"""
Episode Buffer

Stores episodic data for training the neural networks.

Each episode consists of:
1. Sequence of game states
2. Sequence of actions
3. Sequence of rewards  
4. Sequence of MCTS policies
5. Sequence of state evaluations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


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
            policies: List of policy distributions (dicts)
            values: List of state values/evaluations
        """
        episode = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'policies': policies,
            'values': values
        }
        self.episodes.append(episode)
    
    def get_training_data(self):
        """Return all episodes as training data.
        
        Used by NNM for training.
        Returns:
            List of (state, target_return) tuples
        """
        training_data = []
        
        for episode in self.episodes:
            states = episode['states']
            final_value = episode['values'][-1] if episode['values'] else 0.0
            
            for state in states:
                training_data.append((float(state), float(final_value)))
        
        return training_data
    
    def clear(self):
        """Clear all episodes from buffer."""
        self.episodes = []
    
    def size(self):
        """Return number of episodes in buffer."""
        return len(self.episodes)
    
    def __repr__(self):
        return f"EpisodeBuffer(episodes={self.size()})"
