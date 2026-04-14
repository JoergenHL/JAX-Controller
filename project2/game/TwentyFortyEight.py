import math
import numpy as np
import random


class TwentyFortyEight:
    """2048 game.

    State: flat numpy float32 array of shape (16,) — a 4×4 board.
    Each cell stores log2(tile) for occupied cells, 0.0 for empty cells.
        empty = 0.0,  tile 2 = 1.0,  tile 4 = 2.0,  tile 8 = 3.0,
        tile 16 = 4.0, tile 32 = 5.0, ..., tile 2048 = 11.0

    Log2 encoding gives bounded, uniformly-spaced values as NNr input.
    Merging two tiles of value v produces v+1 in log2 space.

    Reward: non-zero only when the current maximum tile advances.
      If the board max is 16 and two 16s merge → 32: reward = 16.
      If the board max is 16 and two 8s merge → 16: reward = 0 (max unchanged).
      No merge: reward = 0.

    Terminal: no valid move in any of the 4 directions.

    is_win: tile ≥ 2048 (log2 ≥ 11) — an unreachable threshold so that the
    binary win rate from evaluate() stays near 0% while average max tile is
    the real progress metric.
    """

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

    @property
    def action_space(self):
        """All possible actions regardless of current board state."""
        return self.ACTIONS

    def __init__(self):
        self.state_dim     = 16
        self.num_actions   = 4
        self.win_tile_log2 = 11.0   # log2(2048) — effectively unreachable
        # Divide reward/value targets by this in rlm._train_networks to keep
        # loss magnitudes comparable. Reward is now log2(merge_score) per step
        # (~2–11 per productive move). A full game has 50–150 productive merges
        # → total return ~150–600. Dividing by 32 keeps value targets in ~5–20.
        self.reward_scale  = 32.0

    # ── Core GSM interface ─────────────────────────────────────────────────────

    def initial_state(self):
        """Empty board with two random starter tiles."""
        state = np.zeros(16, dtype=np.float32)
        state = self._spawn_tile(state)
        state = self._spawn_tile(state)
        return state

    def legal_actions(self, state):
        """Actions that actually change the board (i.e. are valid moves)."""
        if self.is_terminal(state):
            return []
        return [a for a in self.ACTIONS if self._action_changes_board(state, a)]

    def next_state(self, state, action):
        """Apply action: slide/merge, then spawn one new tile."""
        new_state, _, _ = self._apply_move(state, action)
        return self._spawn_tile(new_state)

    def reward(self, state, action, next_state):
        """Reward = log₂(total merge score) for this move.

        Every merge contributes to the score: two tiles of value v merging into
        2v add 2v to the merge score. Taking log₂ normalises across the
        exponentially growing tile values:
            2+2→4:   merge_score=4,    reward=2
            8+8→16:  merge_score=16,   reward=4
            64+64→128: merge_score=128, reward=7

        Dense reward (fires on every merge, ~100–200 times per game) vs. the
        previous max-tile-only reward (~8–10 times per game), giving the agent
        a clear, graded signal for making good vs. wasted moves.
        """
        _, _, merge_score = self._apply_move(state, action)
        if merge_score == 0:
            return 0.0
        return math.log2(merge_score)

    def is_terminal(self, state):
        """No valid moves remain: board is full and no adjacent equal tiles."""
        if np.any(state == 0.0):
            return False
        board = state.reshape(4, 4)
        if np.any(board[:, :-1] == board[:, 1:]):   # adjacent equal horizontally
            return False
        if np.any(board[:-1, :] == board[1:, :]):   # adjacent equal vertically
            return False
        return True

    def is_win(self, state):
        """Tile ≥ 2048 — unreachable threshold used only in rlm.evaluate()."""
        return float(np.max(state)) >= self.win_tile_log2

    def max_tile(self, state):
        """Actual tile value of the largest tile on the board."""
        v = np.max(state)
        return int(2 ** v) if v > 0 else 0

    def render(self, state):
        """Print the 4×4 board with actual tile values."""
        board = state.reshape(4, 4)
        sep = "+--------+--------+--------+--------+"
        print(sep)
        for row in board:
            cells = []
            for v in row:
                cells.append("        " if v == 0 else f"{int(2 ** v):8d}")
            print("|" + "|".join(cells) + "|")
            print(sep)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _spawn_tile(self, state):
        """Spawn a tile (2 with prob 0.9, 4 with prob 0.1) at a random empty cell."""
        empty = np.where(state == 0.0)[0]
        if len(empty) == 0:
            return state
        state = state.copy()
        idx = random.choice(empty.tolist())
        state[idx] = 1.0 if random.random() < 0.9 else 2.0   # log2(2)=1, log2(4)=2
        return state

    def _slide_row_left(self, row):
        """Slide a list of 4 log2-values leftward: remove zeros, merge equals, pad.

        Merging two cells with log2-value v produces a cell with value v+1.
        Only one merge per pair per move (a merged cell cannot merge again).

        Returns:
            (new_row, max_merged_log2, merge_score)
            max_merged_log2 = log2 value of the largest tile created by merging,
                              or 0 if no merge occurred.
            merge_score     = sum of actual tile values created by all merges
                              (standard 2048 scoring), 0 if no merges.
        """
        tiles = [t for t in row if t != 0]
        result = []
        max_merged  = 0
        merge_score = 0
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                new_val = tiles[i] + 1   # log2: merge doubles the tile
                result.append(new_val)
                merge_score += int(2 ** new_val)   # actual value of merged tile
                if new_val > max_merged:
                    max_merged = new_val
                i += 2
            else:
                result.append(tiles[i])
                i += 1
        result += [0] * (4 - len(result))
        return result, max_merged, merge_score

    def _apply_move(self, state, action):
        """Slide and merge the board for the given action. Does NOT spawn a tile.

        Returns:
            (new_flat_state, max_merged_log2, total_score)
            total_score = sum of all merged tile values this move (standard 2048 score).
        """
        board = state.reshape(4, 4)
        max_merged  = 0
        total_score = 0

        if action == "LEFT":
            rows = []
            for row in board:
                new_row, m, s = self._slide_row_left(row.tolist())
                rows.append(new_row)
                max_merged  = max(max_merged, m)
                total_score += s
            new_board = np.array(rows, dtype=np.float32)

        elif action == "RIGHT":
            rows = []
            for row in board:
                new_row, m, s = self._slide_row_left(row[::-1].tolist())
                rows.append(new_row[::-1])
                max_merged  = max(max_merged, m)
                total_score += s
            new_board = np.array(rows, dtype=np.float32)

        elif action == "UP":
            cols = []
            for col in board.T:
                new_col, m, s = self._slide_row_left(col.tolist())
                cols.append(new_col)
                max_merged  = max(max_merged, m)
                total_score += s
            new_board = np.array(cols, dtype=np.float32).T

        elif action == "DOWN":
            cols = []
            for col in board.T:
                new_col, m, s = self._slide_row_left(col[::-1].tolist())
                cols.append(new_col[::-1])
                max_merged  = max(max_merged, m)
                total_score += s
            new_board = np.array(cols, dtype=np.float32).T

        else:
            raise ValueError(f"Unknown action: {action}")

        return new_board.flatten(), max_merged, total_score

    def _action_changes_board(self, state, action):
        """True if applying action produces a different board (move is valid)."""
        new_state, _, _ = self._apply_move(state, action)
        return not np.array_equal(state, new_state)
