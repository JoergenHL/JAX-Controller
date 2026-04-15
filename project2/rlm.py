"""Reinforcement Learning Manager."""

import contextlib
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import jax.numpy as jnp
from mcts.mcts import MCTS
from buffer import EpisodeBuffer
from game.ASM import ASM
import config


class ReinforcementLearningManager:
    """Orchestrates the AlphaZero self-play training loop.

    Responsibilities:
    - Run game episodes using MCTS to choose actions
    - Store episode data in the EpisodeBuffer
    - Trigger network training via NNManager
    - Evaluate win rate
    """

    def __init__(self, game_state_manager, nn_manager):
        self.gsm = game_state_manager
        self.nnm = nn_manager
        self.episode_buffer = EpisodeBuffer(
            max_size=config.training["buffer_size"]
        )
        self.asm = ASM()

        # Stage 4B: u-MCTS operates entirely in abstract space.
        # NNr encodes the real state to σ once at the root.
        # NNd drives all node expansions (no real game calls during search).
        # NNp evaluates leaves and provides policy priors for PUCT.
        self.mcts = MCTS(
            nn_r         = nn_manager.get_net("nnr"),
            nn_d         = nn_manager.get_net("nnd"),
            nn_p         = nn_manager.get_net("nnp"),
            action_space = self.gsm.action_space,
            use_puct     = True,
            dir_alpha    = config.mcts["dir_alpha"],
            dir_epsilon  = config.mcts["dir_epsilon"],
        )
        self.mcts.num_simulations = config.mcts["num_simulations"]
        self.mcts.c     = config.mcts["c"]
        self.mcts.d_max = config.mcts["d_max"]

    # ── Network interface ──────────────────────────────────────────────────────

    def _predict(self, state):
        """Run the NNr → NNp pipeline on a single real game state.

        Used for evaluation and show_predictions() outside of MCTS.
        (Inside u-MCTS, ASM is called directly on abstract states.)

        Returns:
            value: scalar estimate of the state's worth
            policy_logits: raw (pre-softmax) action preferences
        """
        abstract = self.asm.map_abstract_state(state, self.nnm.get_net("nnr"))
        return self.asm.predict(abstract, self.nnm.get_net("nnp"))

    # ── Episode collection ─────────────────────────────────────────────────────

    def collect_episode(self):
        """Play one game using MCTS and return the trajectory.

        At each step:
          1. Run MCTS simulations from the current state
          2. Pick the most-visited action
          3. Record the state, chosen action, reward, and MCTS visit distribution

        After the game ends, compute per-step returns (G_t = sum of future rewards).
        For LineWorld, the reward is sparse (only ±1 at the terminal step), so
        every state in a winning episode gets return +1 and every state in a
        losing episode gets return -1.
        """
        states, actions, rewards, policies, mcts_values = [], [], [], [], []

        state = self.gsm.initial_state()
        max_steps = 500   # safety bound for games without a natural early terminal
        steps = 0
        while not self.gsm.is_terminal(state) and steps < max_steps:
            states.append(state)
            steps += 1

            _, policy, mcts_val = self.mcts.search(state)
            # Store the MCTS root value estimate for n-step bootstrap targets.
            # These replace pure Monte-Carlo returns in _train_networks, cutting
            # value-target variance from ~200-step horizon to n_step horizon.
            mcts_values.append(mcts_val)

            # Sample action from MCTS visit distribution (AlphaZero training convention).
            # Argmax would always pick LEFT when values are uniform (equal visit counts),
            # preventing exploration. Sampling ensures diverse self-play data from the start.
            total = sum(policy.values()) or 1
            action = random.choices(list(policy.keys()),
                                    weights=[policy[a] / total for a in policy],
                                    k=1)[0]

            actions.append(action)
            policies.append(policy)    # {action: visit_count} from MCTS

            next_state = self.gsm.next_state(state, action)
            reward = self.gsm.reward(state, action, next_state)
            rewards.append(reward)

            state = next_state

        return {'states': states, 'actions': actions,
                'rewards': rewards, 'policies': policies, 'values': mcts_values}

    # ── Parallel episode collection ────────────────────────────────────────────

    def _get_network_dims(self) -> dict:
        """Read architecture dims from existing models.

        Returns {name: [in_dim, h1, ..., out_dim]} by inspecting weight shapes.
        Used to recreate the same architecture in worker processes.
        """
        dims = {}
        for name, model in self.nnm.models.items():
            model_dims = [model.layers[0].w.value.shape[0]]
            for layer in model.layers:
                model_dims.append(layer.w.value.shape[1])
            dims[name] = model_dims
        return dims

    def _collect_parallel(self, n: int, pool: ProcessPoolExecutor) -> list:
        """Submit n episode-collection tasks to the worker pool and gather results.

        All n tasks are submitted before waiting on any, so they run truly in
        parallel. The weight snapshot is taken once and shared across all tasks
        in this round (workers receive read-only copies — no write-back).

        Returns a list of n episode dicts (same format as collect_episode()).
        """
        from worker import collect_episode_worker

        args = {
            "game_name":    self.gsm.__class__.__name__,
            "network_dims": self._get_network_dims(),
            "layer_weights": self.nnm.get_layer_weights(),
            "mcts_cfg":     dict(config.mcts),
            "max_steps":    500,
        }
        futures = [pool.submit(collect_episode_worker, args) for _ in range(n)]
        return [f.result() for f in futures]

    # ── Training loop ──────────────────────────────────────────────────────────

    def train(self, num_iterations=None, episodes_per_iter=None, epochs=None):
        """Self-play training loop.

        Each iteration:
          1. Collect episodes using the current network
          2. Train the network on all collected data
          3. Repeat — the improved network generates better data next iteration

        Returns:
            dict with 'losses' (per-epoch history) and 'iter_boundaries'
            (epoch indices where each new iteration starts), for plotting.
        """
        num_iterations    = num_iterations    or config.training["num_iterations"]
        episodes_per_iter = episodes_per_iter or config.training["episodes_per_iter"]
        epochs            = epochs            or config.training["epochs_per_iter"]

        num_workers = config.training.get("num_workers", 1)
        use_parallel = num_workers > 1

        print(f"\n{'='*50}")
        print("Self-play training")
        if use_parallel:
            print(f"  Episode collection: {num_workers} parallel workers")
        print(f"{'='*50}\n")

        all_losses      = []   # (total, value, policy, reward) per epoch, all iterations
        iter_boundaries = []   # epoch index at which each iteration begins
        eval_scores     = []   # (iteration, pct, avg_tile, [tile_per_game]) when enabled

        eval_every = config.viz.get("eval_every", 0)
        eval_games = config.viz.get("eval_games", 5)

        # Create a persistent worker pool (lives for all iterations so JAX
        # initialisation cost in each worker is paid once, not per iteration).
        # Falls back to a no-op context when running sequentially.
        if use_parallel:
            mp_ctx   = multiprocessing.get_context("spawn")
            pool_ctx = ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx)
        else:
            pool_ctx = contextlib.nullcontext()

        with pool_ctx as pool:
            for it in range(num_iterations):
                print(f"[Iteration {it+1}/{num_iterations}]")

                print(f"  Collecting {episodes_per_iter} episodes"
                      f"{' (parallel)' if use_parallel else ''}...")
                if use_parallel:
                    episodes = self._collect_parallel(episodes_per_iter, pool)
                else:
                    episodes = [self.collect_episode()
                                for _ in range(episodes_per_iter)]

                for episode in episodes:
                    self.episode_buffer.add_episode(
                        episode['states'],
                        episode['actions'],
                        episode['rewards'],
                        episode['policies'],
                        episode['values'],
                    )

                print(f"  Training for {epochs} epochs...")
                iter_boundaries.append(len(all_losses))
                iter_losses = self._train_networks(epochs)
                if iter_losses:
                    all_losses.extend(iter_losses)

                print(f"  Buffer: {self.episode_buffer.size()} episodes")

                # Optional in-training evaluation — off by default (eval_every=0)
                if eval_every > 0 and (it + 1) % eval_every == 0:
                    pct, avg_tile, tiles = self.evaluate(
                        num_games=eval_games,
                        pool=pool if use_parallel else None,
                    )
                    eval_scores.append((it + 1, pct, avg_tile, tiles))

                print()

        return {'losses': all_losses, 'iter_boundaries': iter_boundaries,
                'eval_scores': eval_scores}

    def _train_networks(self, epochs):
        """Build BPTT minibatches from the episode buffer and train all three networks.

        Stage 4A: replaces the flat train_repr_pred call with full BPTT roll-ahead.

        For each episode in the buffer, we extract every valid window of
        roll_ahead consecutive steps. Each window provides:
          - Starting real state s_k
          - The next roll_ahead actions (what the agent actually did)
          - Value targets: episode returns G_{k}…G_{k+w-1}
          - Policy targets: MCTS visit distributions π_k…π_{k+w-1}
          - Reward targets: actual immediate rewards r_{k+1}…r_{k+w}

        NNManager.train_bptt then unrolls NNr → NNd^w → NNp through this window
        and backpropagates through the entire unrolled graph.
        """
        if self.episode_buffer.size() == 0:
            return None

        action_order  = self.gsm.action_space
        action_to_idx = {a: i for i, a in enumerate(action_order)}
        roll_ahead    = config.training["roll_ahead"]
        gamma         = config.training.get("gamma", 1.0)

        # Normalize reward/value targets by a game-specific scale so that loss
        # magnitudes stay comparable regardless of reward range (e.g. 2048
        # rewards can be 16+ per merge; LineWorld rewards are ±1).
        scale = self.gsm.reward_scale

        minibatches = []
        for ep_idx in range(self.episode_buffer.size()):
            ep       = self.episode_buffer.get_episode(ep_idx)
            states   = ep['states']
            actions  = ep['actions']
            mcts_vs  = ep['values']   # per-step MCTS value estimates (not MC returns)
            policies = ep['policies']
            rewards  = ep['rewards']
            T        = len(states)

            # N-step bootstrap targets (MuZero Section 3):
            #   z_k = sum(rewards[k:k+n]) / scale  +  v_{k+n}
            # Rewards are raw game values and must be divided by scale.
            # v_{k+n} is the MCTS root value, which is already in normalized
            # units (NNd/NNp outputs are trained on scale-divided targets) —
            # so it must NOT be divided by scale again.
            # ^ not used anymore, loss exploded
            mc_returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                mc_returns.insert(0, G / scale)
            

            # Each valid starting step k needs roll_ahead more steps after it
            for k in range(T - roll_ahead + 1):
                p_arrays = []
                for pt in policies[k : k + roll_ahead]:
                    total = sum(pt.values()) or 1
                    p_arrays.append([pt.get(a, 0.0) / total for a in action_order])

                minibatches.append({
                    'state':          np.array(states[k], dtype=np.float32).flatten(),
                    'action_indices': [action_to_idx[a] for a in actions[k : k + roll_ahead]],
                    'value_targets': [mc_returns[k + j] for j in range(roll_ahead)],
                    'policy_targets': p_arrays,
                    'reward_targets': [float(r) / scale for r in rewards[k : k + roll_ahead]],
                })

        if not minibatches:
            return None

        return self.nnm.train_bptt(
            minibatches,
            abstract_dim=config.nn["abstract_dim"],
            num_actions=self.gsm.num_actions,
            num_epochs=epochs,
            learning_rate=config.nn["learning_rate"],
        )

    # ── Policy data sampling ───────────────────────────────────────────────────

    def sample_policy_data(self, num_games: int = 20) -> dict:
        """Play num_games greedy games and collect per-step policy/value data.

        No MCTS — NNr+NNp forward passes only. Fast (<1s for 20 games).
        Intended for post-training policy analysis visualizations, not during
        the training loop.

        Returns a dict with numpy arrays of shape [N, ...]:
            action_space: list of action strings (from game)
            probs:        softmax action probabilities  [N, num_actions]
            values:       NNp value estimates           [N]
            max_tiles:    game.max_tile() at each step  [N]
        """
        from flax import nnx
        import jax
        _net_fwd = nnx.jit(lambda model, x: model(x))

        nn_r         = self.nnm.get_net("nnr")
        nn_p         = self.nnm.get_net("nnp")
        action_space = self.gsm.action_space

        all_probs     = []
        all_values    = []
        all_max_tiles = []

        for _ in range(num_games):
            state = self.gsm.initial_state()
            steps = 0
            while not self.gsm.is_terminal(state) and steps < 500:
                sigma  = _net_fwd(nn_r, jnp.atleast_2d(
                    jnp.array(state, dtype=jnp.float32)
                ))
                output = _net_fwd(nn_p, sigma)[0]

                value = float(output[0])
                probs = np.array(jax.nn.softmax(output[1:]))

                all_values.append(value)
                all_probs.append(probs)
                all_max_tiles.append(self.gsm.max_tile(state))

                action = action_space[int(jnp.argmax(output[1:]))]
                state  = self.gsm.next_state(state, action)
                steps += 1

        return {
            "action_space": action_space,
            "probs":        np.array(all_probs,     dtype=np.float32),
            "values":       np.array(all_values,    dtype=np.float32),
            "max_tiles":    np.array(all_max_tiles, dtype=np.float32),
        }

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, num_games=10, pool=None):
        """Play num_games using greedy NNr → NNp (no MCTS, no NNd).

        Matches run_agent.py exactly — this is what the deployed agent does.
        Runs in parallel when num_workers > 1 and a pool is supplied (or
        a temporary pool is created internally).

        Returns:
            pct:       win rate (0–100)
            avg_tile:  average max tile across all games
            max_tiles: list of max tile per game (length = num_games)
        """
        from worker import evaluate_greedy_worker

        num_workers = config.training.get("num_workers", 1)
        use_parallel = num_workers > 1

        print(f"\n  Evaluating greedy NNr+NNp ({num_games} games"
              f"{', parallel' if use_parallel else ''})...")

        args = {
            "game_name":    self.gsm.__class__.__name__,
            "network_dims": {k: v for k, v in self._get_network_dims().items()
                             if k in ("nnr", "nnp")},
            "layer_weights": {k: v for k, v in self.nnm.get_layer_weights().items()
                              if k in ("nnr", "nnp")},
            "num_games":    num_games,
            "max_steps":    500,
        }

        if use_parallel:
            # Split games evenly across workers; last worker gets the remainder.
            games_per_worker = num_games // num_workers
            remainder        = num_games % num_workers
            splits = [games_per_worker + (1 if i < remainder else 0)
                      for i in range(num_workers)]
            splits = [s for s in splits if s > 0]

            def _run(pool_obj):
                worker_args = [{**args, "num_games": n} for n in splits]
                futures = [pool_obj.submit(evaluate_greedy_worker, a)
                           for a in worker_args]
                wins      = 0
                max_tiles = []
                for f in futures:
                    r = f.result()
                    wins      += r["wins"]
                    max_tiles += r["max_tiles"]
                return wins, max_tiles

            if pool is not None:
                wins, max_tiles = _run(pool)
            else:
                mp_ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=num_workers,
                                         mp_context=mp_ctx) as tmp_pool:
                    wins, max_tiles = _run(tmp_pool)
        else:
            result   = evaluate_greedy_worker(args)
            wins      = result["wins"]
            max_tiles = result["max_tiles"]

        pct      = 100 * wins / num_games
        avg_tile = sum(max_tiles) / len(max_tiles)
        best     = max(max_tiles)
        print(f"  Tiles: {max_tiles}  avg={avg_tile:.0f}  best={best}")
        return pct, avg_tile, max_tiles

    def evaluate_mcts(self, num_games=100):
        """Play num_games using the current MCTS+network policy (NNr+NNd+NNp).

        Slower than evaluate() — use for an upper-bound comparison of how
        much planning adds on top of the greedy policy.

        Returns:
            pct:       win rate (0–100)
            avg_tile:  average max tile across all games
            max_tiles: list of max tile per game (length = num_games)
        """
        print(f"\n  Evaluating MCTS ({num_games} games)...")
        wins = 0
        max_tiles = []
        for _ in range(num_games):
            state = self.gsm.initial_state()
            steps = 0
            while not self.gsm.is_terminal(state) and steps < 300:
                action, _, _ = self.mcts.search(state)
                state = self.gsm.next_state(state, action)
                steps += 1
            if self.gsm.is_win(state):
                wins += 1
            max_tiles.append(self.gsm.max_tile(state))
        pct      = 100 * wins / num_games
        avg_tile = sum(max_tiles) / len(max_tiles)
        best     = max(max_tiles)
        print(f"  Tiles: {max_tiles}  avg={avg_tile:.0f}  best={best}")
        return pct, avg_tile, max_tiles
