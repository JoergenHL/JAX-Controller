"""Reinforcement Learning Manager."""

import random
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
        self.episode_buffer = EpisodeBuffer()
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
        states, actions, rewards, policies = [], [], [], []

        state = self.gsm.initial_state()
        max_steps = 500   # safety bound for games without a natural early terminal
        steps = 0
        while not self.gsm.is_terminal(state) and steps < max_steps:
            states.append(state)
            steps += 1

            _, policy, _ = self.mcts.search(state)

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

        # G_t = r_t + r_{t+1} + … + r_T  (gamma = 1, no discounting)
        returns, G = [], 0
        for r in reversed(rewards):
            G += r
            returns.insert(0, G)

        return {'states': states, 'actions': actions,
                'rewards': rewards, 'policies': policies, 'returns': returns}

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
        num_iterations   = num_iterations   or config.training["num_iterations"]
        episodes_per_iter = episodes_per_iter or config.training["episodes_per_iter"]
        epochs           = epochs           or config.training["epochs_per_iter"]

        print(f"\n{'='*50}")
        print("Self-play training")
        print(f"{'='*50}\n")

        all_losses      = []   # (total, value, policy, reward) per epoch, all iterations
        iter_boundaries = []   # epoch index at which each iteration begins
        eval_scores     = []   # (iteration, pct, avg_tile, [tile_per_game]) when enabled

        eval_every = config.viz.get("eval_every", 0)
        eval_games = config.viz.get("eval_games", 5)

        for it in range(num_iterations):
            print(f"[Iteration {it+1}/{num_iterations}]")

            print(f"  Collecting {episodes_per_iter} episodes...")
            for _ in range(episodes_per_iter):
                episode = self.collect_episode()
                self.episode_buffer.add_episode(
                    episode['states'],
                    episode['actions'],
                    episode['rewards'],
                    episode['policies'],
                    episode['returns'],
                )

            print(f"  Training for {epochs} epochs...")
            iter_boundaries.append(len(all_losses))
            iter_losses = self._train_networks(epochs)
            if iter_losses:
                all_losses.extend(iter_losses)

            print(f"  Buffer: {self.episode_buffer.size()} episodes")

            # Optional in-training evaluation — off by default (eval_every=0)
            if eval_every > 0 and (it + 1) % eval_every == 0:
                pct, avg_tile, tiles = self.evaluate(num_games=eval_games)
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

        # Normalize reward/value targets by a game-specific scale so that loss
        # magnitudes stay comparable regardless of reward range (e.g. 2048
        # rewards can be 16+ per merge; LineWorld rewards are ±1).
        scale = self.gsm.reward_scale

        minibatches = []
        for ep_idx in range(self.episode_buffer.size()):
            ep      = self.episode_buffer.get_episode(ep_idx)
            states  = ep['states']
            actions = ep['actions']
            returns = ep['values']    # per-step returns G_t
            policies = ep['policies']
            rewards  = ep['rewards']

            # Each valid starting step k needs roll_ahead more steps after it
            for k in range(len(states) - roll_ahead + 1):
                p_arrays = []
                for pt in policies[k : k + roll_ahead]:
                    total = sum(pt.values()) or 1
                    p_arrays.append([pt.get(a, 0.0) / total for a in action_order])

                minibatches.append({
                    'state':          np.array(states[k], dtype=np.float32).flatten(),
                    'action_indices': [action_to_idx[a] for a in actions[k : k + roll_ahead]],
                    'value_targets':  [float(v) / scale for v in returns[k : k + roll_ahead]],
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

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, num_games=10):
        """Play num_games using the current MCTS+network policy.

        Returns:
            pct:       win rate (0–100)
            avg_tile:  average max tile across all games
            max_tiles: list of max tile per game (length = num_games)
        """
        print(f"\n  Evaluating ({num_games} games)...")
        wins = 0
        max_tiles = []
        for _ in range(num_games):
            state = self.gsm.initial_state()
            steps = 0
            while not self.gsm.is_terminal(state) and steps < 500:
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
