"""
Episode Buffer

Stores episodic data for training the neural networks.

Each episode consists of:
1. Sequence of game states
2. Sequence of actions
3. Sequence of rewards
4. Sequence of MCTS policies (visit distributions)
5. Sequence of per-step returns (value targets)
"""


class EpisodeBuffer:
    """Stores episodes for training, with a fixed maximum capacity.

    Once the buffer is full, the oldest episode is discarded when a new one
    arrives (FIFO). Keeping the buffer size constant is important for two
    reasons:
      1. Training time per iteration stays constant (batch size doesn't grow).
      2. JAX can reuse its JIT-compiled kernel across iterations because the
         input array shapes don't change.

    A buffer of ~30 episodes ≈ the last 10 iterations of self-play data,
    which provides diverse training samples while keeping recent experience
    more prevalent.
    """

    def __init__(self, max_size: int):
        """
        Args:
            max_size: maximum number of episodes to keep (oldest discarded first).
        """
        self.episodes = []
        self.max_size = max_size

    def add_episode(self, states, actions, rewards, policies, values):
        """Add a complete episode; drop the oldest if the buffer is full."""
        episode = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'policies': policies,  # {action: visit_count}
            'values': values
        }
        if len(self.episodes) >= self.max_size:
            self.episodes.pop(0)   # discard oldest
        self.episodes.append(episode)
    
    def get_episode(self, idx):
        """Return a full episode dict by index.

        Used for BPTT minibatch sampling (PDF: DO_BPTT_TRAINING).
        Each dict contains: states, actions, rewards, policies, values.
        """
        return self.episodes[idx]

    def clear(self):
        """Clear all episodes from buffer."""
        self.episodes = []
    
    def size(self):
        """Return number of episodes in buffer."""
        return len(self.episodes)
    
    def __repr__(self):
        return f"EpisodeBuffer(episodes={self.size()})"
