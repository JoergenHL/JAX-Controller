"""Random baseline: plays games with uniformly random legal actions.

Used to establish a lower bound on performance. If the trained agent doesn't
beat the random baseline, it has not yet learned anything useful.

The evaluate() signature matches ReinforcementLearningManager.evaluate() so
results can be passed directly to the same logging and visualization code.
"""

import random


class RandomBaseline:
    """Plays games by picking uniformly random legal actions at each step."""

    def __init__(self, game):
        self.gsm = game

    def evaluate(self, num_games=20):
        """Play num_games with random actions.

        Returns:
            pct:       win rate (0–100)  — matches rlm.evaluate() signature
            avg_tile:  average max tile across all games
            max_tiles: list of max tile per game (length = num_games)
        """
        print(f"\n  [Baseline] Evaluating ({num_games} random games)...")
        wins, max_tiles = 0, []

        for _ in range(num_games):
            state = self.gsm.initial_state()
            steps = 0
            while not self.gsm.is_terminal(state) and steps < 500:
                actions = self.gsm.legal_actions(state)
                if not actions:
                    break
                state = self.gsm.next_state(state, random.choice(actions))
                steps += 1
            if self.gsm.is_win(state):
                wins += 1
            max_tiles.append(self.gsm.max_tile(state))

        pct      = 100 * wins / num_games
        avg_tile = sum(max_tiles) / len(max_tiles)
        best     = max(max_tiles)
        print(f"  [Baseline] Tiles: {max_tiles}")
        print(f"  [Baseline] avg={avg_tile:.0f}  best={best}  wins={wins}/{num_games}")
        return pct, avg_tile, max_tiles
