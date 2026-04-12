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
    """Stores episodes for training."""
    
    def __init__(self):
        """Initialize empty episode buffer."""
        self.episodes = []
    
    def add_episode(self, states, actions, rewards, policies, values):
        """Add a complete episode to the buffer.
        
        Args:
            states: List of game states visited
            actions: List of actions taken (indices)
            rewards: List of rewards received
            policies: List of policy distributions (dicts with visit counts)
            values: List of state values/evaluations
        """
        episode = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'policies': policies,  # {action: visit_count}
            'values': values
        }
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
