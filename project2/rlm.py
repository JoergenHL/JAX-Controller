"""Reinforcement Learning Manager."""

import contextlib
import glob
import json
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import jax.numpy as jnp
from mcts.mcts import MCTS
from buffer import EpisodeBuffer
from game.ASM import ASM
import config


def compute_sampling_tau(iteration: int, total_iterations: int, cfg: dict) -> float:
    """Compute the self-play action-sampling temperature at a given iteration.

    Shared by the main process (rlm.collect_episode) and the workers
    (worker.collect_episode_worker) so both agree on τ without duplicated logic.

    Schedule:
      iteration < transition_at * total_iterations  → τ = start
      otherwise → linear interp start → end across the remaining iterations.
    """
    start = float(cfg.get("start",         1.0))
    end   = float(cfg.get("end",           0.2))
    frac  = float(cfg.get("transition_at", 0.5))

    if total_iterations <= 1:
        return start

    transition_iter = frac * (total_iterations - 1)
    if iteration <= transition_iter:
        return start

    span = (total_iterations - 1) - transition_iter
    if span <= 0:
        return end
    progress = (iteration - transition_iter) / span
    progress = max(0.0, min(1.0, progress))
    return start + (end - start) * progress


def temperature_sample(policy: dict, tau: float, rng=None):
    """Sample one action from a visit-count dict under temperature τ.

    policy: {action: visit_count}. rng: optional random.Random for determinism.
    """
    import random as _random
    rng = rng or _random
    if tau <= 1e-6:
        # Argmax (ties broken by rng.choice over winners)
        max_v = max(policy.values())
        winners = [a for a, v in policy.items() if v == max_v]
        return rng.choice(winners) if len(winners) > 1 else winners[0]

    inv = 1.0 / tau
    actions, weights = [], []
    for a, v in policy.items():
        actions.append(a)
        weights.append(max(float(v), 0.0) ** inv)
    total = sum(weights) or 1.0
    probs = [w / total for w in weights]
    return rng.choices(actions, weights=probs, k=1)[0]


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
            max_size=self._resolve_buffer_cap()
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

    # ── Buffer sizing ──────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_buffer_cap() -> int:
        """Resolve the replay-buffer capacity from config.

        Priority:
          1. training["buffer_size"] — explicit raw cap (hparam scripts use this)
          2. derived: episodes_per_iter * min(buffer_history_iters, num_iterations)
        """
        explicit = config.training.get("buffer_size")
        if explicit is not None:
            return int(explicit)

        eps      = config.training["episodes_per_iter"]
        hist     = config.training["buffer_history_iters"]
        n_iter   = config.training["num_iterations"]
        return eps * min(hist, n_iter)

    # ── Network interface ──────────────────────────────────────────────────────

    def _predict(self, state):
        """Run the NNr → NNp pipeline on a single real game state.

        Used for evaluation and show_predictions() outside of MCTS.
        (Inside u-MCTS, ASM is called directly on abstract states.)
        No history available here — pads with the state itself (q copies).

        Returns:
            value: scalar estimate of the state's worth
            policy_logits: raw (pre-softmax) action preferences
        """
        q = config.nn.get("q", 0)
        nnr_input = ASM.build_state_window([state], q)
        abstract = self.asm.map_abstract_state(nnr_input, self.nnm.get_net("nnr"))
        return self.asm.predict(abstract, self.nnm.get_net("nnp"))

    # ── Episode collection ─────────────────────────────────────────────────────

    def collect_episode(self, iteration: int = 0, total_iterations: int = 1):
        """Play one game using MCTS and return the trajectory.

        At each step:
          1. Run MCTS simulations from the current state
          2. Sample an action under the current temperature τ
          3. Record the state, chosen action, reward, and MCTS visit distribution

        τ follows the schedule in config.training["sampling_temp"] — high τ
        early in training (exploration), decays toward τ_end late (exploitation).

        After the game ends, compute per-step returns (G_t = sum of future rewards).
        """
        tau = compute_sampling_tau(
            iteration, total_iterations,
            config.training.get("sampling_temp", {}),
        )
        states, actions, rewards, policies, mcts_values = [], [], [], [], []

        state = self.gsm.initial_state()
        max_steps = 500   # safety bound for games without a natural early terminal
        steps = 0
        q = config.nn.get("q", 0)
        state_history = []
        while not self.gsm.is_terminal(state) and steps < max_steps:
            states.append(state)
            state_history.append(state)
            steps += 1

            nnr_input = ASM.build_state_window(state_history, q)
            _, policy, mcts_val = self.mcts.search(nnr_input)
            # Store the MCTS root value estimate for n-step bootstrap targets.
            # These replace pure Monte-Carlo returns in _train_networks, cutting
            # value-target variance from ~200-step horizon to n_step horizon.
            mcts_values.append(mcts_val)

            # Sample action from MCTS visit distribution under temperature τ.
            # τ=1 → proportional to visits (exploration); τ→0 → argmax (exploitation).
            # Restrict to legal actions so the agent never wastes a step on a no-op move.
            legal = set(self.gsm.legal_actions(state))
            legal_policy = {a: v for a, v in policy.items() if a in legal}
            if not legal_policy:
                legal_policy = policy   # fallback: shouldn't happen (is_terminal guard above)
            action = temperature_sample(legal_policy, tau)

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

    def _collect_parallel(self, n: int, pool: ProcessPoolExecutor,
                          iteration: int = 0, total_iterations: int = 1) -> list:
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
            "q":            config.nn.get("q", 0),
            "iteration":          iteration,
            "total_iterations":   total_iterations,
            "sampling_temp_cfg":  dict(config.training.get("sampling_temp", {})),
        }
        futures = [pool.submit(collect_episode_worker, args) for _ in range(n)]
        return [f.result() for f in futures]

    # ── Checkpoint / leaderboard helpers ───────────────────────────────────────

    def _snapshot_metadata(self, network_dims: dict, game_name: str,
                            iteration: int, eval_avg: float,
                            eval_scores: list, kind: str) -> dict:
        """Build the companion JSON dict for a saved .pkl checkpoint.

        Matches the shape that watch_cartpole.py / best_cartpole.py expect:
        they read config.nn (for network reconstruction) and game. The extra
        fields (iteration, eval_avg, kind) are diagnostic — ignored by the
        loaders but useful when inspecting `runs/`.
        """
        return {
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "game":          game_name,
            "reward_scale":  self.gsm.reward_scale,
            "kind":          kind,          # "checkpoint" or "best"
            "iteration":     iteration,
            "eval_avg":      round(float(eval_avg), 2),
            "eval_scores":   list(eval_scores),
            "config": {
                "mcts":     dict(config.mcts),
                "nn":       dict(config.nn),
                "training": dict(config.training),
                "viz":      dict(config.viz),
            },
            "network_dims":  network_dims,
            "iterations":    [],   # empty placeholder for display-only loaders
        }

    def _save_checkpoint(self, pkl_path: str, json_path: str, meta: dict):
        """Persist current nnm weights + companion JSON metadata."""
        self.nnm.save(pkl_path)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _write_leaderboard_files(self, leaderboard: list, run_prefix: str,
                                  network_dims: dict, game_name: str):
        """Rewrite the top-K _bestN.pkl / .json files from in-memory snapshots.

        Called after every leaderboard update. K is small (≤5), pkl writes are
        fast, and the sort order is authoritative — simplest to delete and
        rewrite rather than track renames.
        """
        from nn.NNManager import NNManager
        # Clear any stale _bestN.* files (in case K shrinks or prefix reused)
        for stale in glob.glob(f"{run_prefix}_best*.pkl") + \
                     glob.glob(f"{run_prefix}_best*.json"):
            try:
                os.remove(stale)
            except OSError:
                pass

        for rank, entry in enumerate(leaderboard, start=1):
            pkl_path  = f"{run_prefix}_best{rank}.pkl"
            json_path = f"{run_prefix}_best{rank}.json"

            # Reconstruct a throwaway NNManager with this entry's weights so
            # we can reuse NNManager.save (which produces the (gdef, state)
            # pickle format that watch_cartpole/best_cartpole load from).
            tmp = NNManager()
            for name, dims in network_dims.items():
                tmp.create_net(name, dims)
            tmp.set_layer_weights(entry["layer_weights"])
            tmp.save(pkl_path)
            entry["pkl_path"] = pkl_path
            entry["rank"]     = rank

            meta = self._snapshot_metadata(
                network_dims, game_name,
                iteration=entry["iteration"],
                eval_avg=entry["eval_avg"],
                eval_scores=entry["eval_scores"],
                kind="best",
            )
            meta["rank"] = rank
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)

    def _update_leaderboard(self, leaderboard: list, iteration: int,
                             eval_avg: float, eval_scores: list,
                             run_prefix: str, network_dims: dict,
                             game_name: str, k: int, threshold: float) -> list:
        """Insert a candidate into the top-K leaderboard if it qualifies.

        Returns the (possibly modified) leaderboard sorted best-first.
        """
        if k <= 0 or eval_avg < threshold:
            return leaderboard

        # Fast reject: already full and strictly worse than the worst kept entry
        if len(leaderboard) >= k and eval_avg <= leaderboard[-1]["eval_avg"]:
            return leaderboard

        leaderboard.append({
            "iteration":     iteration,
            "eval_avg":      float(eval_avg),
            "eval_scores":   list(eval_scores),
            "layer_weights": self.nnm.get_layer_weights(),
        })
        leaderboard.sort(key=lambda e: e["eval_avg"], reverse=True)
        del leaderboard[k:]

        self._write_leaderboard_files(leaderboard, run_prefix,
                                       network_dims, game_name)
        new_rank = next((i for i, e in enumerate(leaderboard)
                         if e["iteration"] == iteration), None)
        if new_rank is not None:
            print(f"  ★ Leaderboard update: iter {iteration} entered at rank "
                  f"{new_rank + 1} (eval_avg={eval_avg:.1f})")
        return leaderboard

    # ── Training loop ──────────────────────────────────────────────────────────

    def train(self, num_iterations=None, episodes_per_iter=None,
              updates_per_iter=None, minibatch_size=None,
              run_prefix: str = None, network_dims: dict = None,
              game_name: str = None):
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
        updates_per_iter  = updates_per_iter  or config.training["updates_per_iter"]
        minibatch_size    = minibatch_size    or config.training["minibatch_size"]

        # Total gradient steps across the entire run — drives the cosine LR
        # schedule inside NNManager. Computed once; same value passed every call
        # since optax's schedule tracks its own step count in opt_state.
        total_updates = num_iterations * updates_per_iter

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
        leaderboard     = []   # top-K entries, sorted best-first

        eval_every = config.viz.get("eval_every", 0)
        eval_games = config.viz.get("eval_games", 5)

        # Checkpoint + leaderboard setup. File persistence is gated on run_prefix;
        # hparam-search callers pass None to skip all disk writes.
        ckpt_every  = config.viz.get("checkpoint_every", 0) if run_prefix else 0
        lb_k        = config.viz.get("best_leaderboard_k", 0) if run_prefix else 0
        lb_thresh   = config.viz.get("best_threshold", 0.0)
        persist     = run_prefix is not None and network_dims is not None
        if persist and game_name is None:
            game_name = self.gsm.__class__.__name__

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

                tau_now = compute_sampling_tau(
                    it, num_iterations,
                    config.training.get("sampling_temp", {}),
                )
                print(f"  Collecting {episodes_per_iter} episodes"
                      f"{' (parallel)' if use_parallel else ''}"
                      f"  τ={tau_now:.2f}")
                if use_parallel:
                    episodes = self._collect_parallel(
                        episodes_per_iter, pool,
                        iteration=it, total_iterations=num_iterations,
                    )
                else:
                    episodes = [self.collect_episode(
                                    iteration=it,
                                    total_iterations=num_iterations,
                                )
                                for _ in range(episodes_per_iter)]

                for episode in episodes:
                    self.episode_buffer.add_episode(
                        episode['states'],
                        episode['actions'],
                        episode['rewards'],
                        episode['policies'],
                        episode['values'],
                    )

                print(f"  Training for {updates_per_iter} updates (mbs={minibatch_size})...")
                iter_boundaries.append(len(all_losses))
                iter_losses = self._train_networks(updates_per_iter, minibatch_size,
                                                    total_updates=total_updates)
                if iter_losses:
                    all_losses.extend(iter_losses)

                print(f"  Buffer: {self.episode_buffer.size()} episodes")

                # Optional in-training evaluation — off by default (eval_every=0)
                this_iter_eval = None
                if eval_every > 0 and (it + 1) % eval_every == 0:
                    pct, avg_tile, tiles = self.evaluate(
                        num_games=eval_games,
                        pool=pool if use_parallel else None,
                    )
                    eval_scores.append((it + 1, pct, avg_tile, tiles))
                    this_iter_eval = (avg_tile, tiles)

                # Interval checkpoints (video progression snapshots)
                if persist and ckpt_every > 0 and (it + 1) % ckpt_every == 0:
                    ckpt_pkl  = f"{run_prefix}_ckpt_iter{it + 1:03d}.pkl"
                    ckpt_json = f"{run_prefix}_ckpt_iter{it + 1:03d}.json"
                    avg_for_meta    = this_iter_eval[0] if this_iter_eval else 0.0
                    scores_for_meta = this_iter_eval[1] if this_iter_eval else []
                    meta = self._snapshot_metadata(
                        network_dims, game_name,
                        iteration=it + 1,
                        eval_avg=avg_for_meta,
                        eval_scores=scores_for_meta,
                        kind="checkpoint",
                    )
                    self._save_checkpoint(ckpt_pkl, ckpt_json, meta)

                # Top-K leaderboard update (only when this iteration was eval'd)
                if persist and lb_k > 0 and this_iter_eval is not None:
                    leaderboard = self._update_leaderboard(
                        leaderboard,
                        iteration=it + 1,
                        eval_avg=this_iter_eval[0],
                        eval_scores=this_iter_eval[1],
                        run_prefix=run_prefix,
                        network_dims=network_dims,
                        game_name=game_name,
                        k=lb_k,
                        threshold=lb_thresh,
                    )

                print()

        return {'losses': all_losses, 'iter_boundaries': iter_boundaries,
                'eval_scores': eval_scores, 'leaderboard': leaderboard}

    def _train_networks(self, updates_per_iter, minibatch_size,
                        total_updates: int = None):
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
        q             = config.nn.get("q", 0)

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

                nnr_input = ASM.build_state_window(states[max(0, k - q): k + 1], q)
                minibatches.append({
                    'state':          nnr_input,
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
            num_updates=updates_per_iter,
            minibatch_size=minibatch_size,
            learning_rate=config.nn["learning_rate"],
            total_updates=total_updates,
            lr_schedule=config.nn.get("lr_schedule", "constant"),
            lr_floor_frac=config.nn.get("lr_floor_frac", 0.1),
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

        q = config.nn.get("q", 0)
        for _ in range(num_games):
            state = self.gsm.initial_state()
            state_history = []
            steps = 0
            while not self.gsm.is_terminal(state) and steps < 500:
                state_history.append(state)
                nnr_input = ASM.build_state_window(state_history, q)
                sigma  = _net_fwd(nn_r, jnp.atleast_2d(
                    jnp.array(nnr_input, dtype=jnp.float32)
                ))
                output = _net_fwd(nn_p, sigma)[0]

                value = float(output[0])
                probs = np.array(jax.nn.softmax(output[1:]))

                all_values.append(value)
                all_probs.append(probs)
                all_max_tiles.append(self.gsm.eval_score(steps, state))

                legal = self.gsm.legal_actions(state)
                logits = np.array(output[1:])
                logits[[i for i, a in enumerate(action_space) if a not in legal]] = float('-inf')
                action = action_space[int(np.argmax(logits))]
                state  = self.gsm.next_state(state, action)
                steps += 1

        return {
            "action_space": action_space,
            "probs":        np.array(all_probs,     dtype=np.float32),
            "values":       np.array(all_values,    dtype=np.float32),
            "scores":       np.array(all_max_tiles, dtype=np.float32),
        }

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, num_games=10, pool=None):
        """Play num_games using greedy NNr → NNp (no MCTS, no NNd).

        Matches run_agent.py exactly — this is what the deployed agent does.
        Runs in parallel when num_workers > 1 and a pool is supplied (or
        a temporary pool is created internally).

        Returns:
            pct:    win rate (0–100)
            avg:    average eval_score across all games
            scores: list of eval_score per game (length = num_games)
        """
        from worker import evaluate_greedy_worker

        num_workers = config.training.get("num_workers", 1)
        use_parallel = num_workers > 1
        label = getattr(self.gsm, "score_label", "Score")

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
            "q":            config.nn.get("q", 0),
        }

        if use_parallel:
            games_per_worker = num_games // num_workers
            remainder        = num_games % num_workers
            splits = [games_per_worker + (1 if i < remainder else 0)
                      for i in range(num_workers)]
            splits = [s for s in splits if s > 0]

            def _run(pool_obj):
                worker_args = [{**args, "num_games": n} for n in splits]
                futures = [pool_obj.submit(evaluate_greedy_worker, a)
                           for a in worker_args]
                wins, scores = 0, []
                for f in futures:
                    r = f.result()
                    wins   += r["wins"]
                    scores += r["scores"]
                return wins, scores

            if pool is not None:
                wins, scores = _run(pool)
            else:
                mp_ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=num_workers,
                                         mp_context=mp_ctx) as tmp_pool:
                    wins, scores = _run(tmp_pool)
        else:
            result = evaluate_greedy_worker(args)
            wins   = result["wins"]
            scores = result["scores"]

        pct  = 100 * wins / num_games
        avg  = sum(scores) / len(scores)
        best = max(scores)
        print(f"  {label}: {scores}  avg={avg:.0f}  best={best}")
        return pct, avg, scores

    def evaluate_mcts(self, num_games=100):
        """Play num_games using the current MCTS+network policy (NNr+NNd+NNp).

        Slower than evaluate() — use for an upper-bound comparison of how
        much planning adds on top of the greedy policy.

        Returns:
            pct:    win rate (0–100)
            avg:    average eval_score across all games
            scores: list of eval_score per game (length = num_games)
        """
        label = getattr(self.gsm, "score_label", "Score")
        print(f"\n  Evaluating MCTS ({num_games} games)...")
        q = config.nn.get("q", 0)
        wins, scores = 0, []
        for _ in range(num_games):
            state = self.gsm.initial_state()
            state_history = []
            steps = 0
            while not self.gsm.is_terminal(state) and steps < 300:
                state_history.append(state)
                nnr_input = ASM.build_state_window(state_history, q)
                action, _, _ = self.mcts.search(nnr_input)
                state = self.gsm.next_state(state, action)
                steps += 1
            if self.gsm.is_win(state):
                wins += 1
            scores.append(self.gsm.eval_score(steps, state))
        pct  = 100 * wins / num_games
        avg  = sum(scores) / len(scores)
        best = max(scores)
        print(f"  {label}: {scores}  avg={avg:.0f}  best={best}")
        return pct, avg, scores
