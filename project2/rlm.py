"""Reinforcement Learning Manager."""

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

        # Stage 3: two networks instead of one combined network.
        # NNr maps real states → abstract states.
        # NNp maps abstract states → (value, policy).
        # _predict chains them: real_state → ASM.map_abstract_state → ASM.predict.
        # MCTS still operates on real game states (GSM drives transitions);
        # only the leaf evaluation uses the abstract-state pipeline.
        self.mcts = MCTS(self.gsm, num_actions=2,
                         nn_pred=self._predict, use_puct=True)
        self.mcts.num_simulations = config.mcts["num_simulations"]
        self.mcts.c = config.mcts["c"]

    # ── Network interface ──────────────────────────────────────────────────────

    def _predict(self, state):
        """Run the NNr → NNp pipeline on a single real game state.

        Stage 3 change: instead of one network directly mapping real_state → output,
        we now go real_state → NNr → abstract_state → NNp → (value, policy_logits).
        NNp never sees the raw game state — only the abstract representation.

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
        while not self.gsm.is_terminal(state):
            states.append(state)

            action, policy, _ = self.mcts.search(state)
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
        print("AlphaZero self-play training")
        print(f"{'='*50}\n")

        all_losses = []       # (total, value, policy) per epoch, all iterations
        iter_boundaries = []  # epoch index at which each iteration begins

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

            print(f"  Buffer: {self.episode_buffer.size()} episodes\n")

        return {'losses': all_losses, 'iter_boundaries': iter_boundaries}

    def _train_networks(self, epochs):
        """Extract training targets from the buffer and run one training pass.

        Value target:  the actual episode return G_t (what happened)
        Policy target: MCTS visit distribution π (what the search preferred)

        The network learns to imitate what MCTS does, and MCTS improves as
        the network's estimates become more accurate — the bootstrap cycle.
        """
        training_data, policy_targets = self.episode_buffer.get_training_data_with_policies()
        if not training_data:
            return None

        states        = [s for s, _ in training_data]
        value_targets = [v for _, v in training_data]

        # Map visit-count dicts → probability arrays in canonical action order
        action_order = self.gsm.legal_actions(self.gsm.initial_state())
        policy_arrays = [
            [pt.get(a, 0.0) for a in action_order]
            for pt in policy_targets
        ]

        return self.nnm.train_repr_pred(
            states, value_targets, policy_arrays,
            num_epochs=epochs,
            learning_rate=config.nn["learning_rate"],
        )

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, num_games=20):
        """Play num_games using the current MCTS+network policy and report win rate."""
        print(f"\n  Evaluating ({num_games} games)...")
        wins = 0
        for _ in range(num_games):
            state = self.gsm.initial_state()
            while not self.gsm.is_terminal(state):
                action, _, _ = self.mcts.search(state)
                state = self.gsm.next_state(state, action)
            if state == self.gsm.max_position:
                wins += 1
        pct = 100 * wins / num_games
        print(f"  Win rate: {wins}/{num_games} ({pct:.0f}%)")
        return pct
