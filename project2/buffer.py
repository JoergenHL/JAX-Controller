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
    
    def get_training_data_with_policies(self):
        """Return training data WITH policy targets (for Stage 2B+).
        
        Returns:
            Tuple of (training_data, policy_targets) where:
            - training_data: [(state, target_return), ...]
            - policy_targets: [{action: normalized_prob, ...}, ...]
        """
        training_data = []
        policy_targets = []
        
        for episode in self.episodes:
            states   = episode['states']
            values   = episode['values']   # one return per step
            policies = episode['policies']

            for state, value, policy_dict in zip(states, values, policies):
                training_data.append((float(state), float(value)))
                
                # Normalize visit counts to probabilities
                if policy_dict:
                    total_visits = sum(policy_dict.values())
                    normalized_policy = {
                        action: float(count) / total_visits 
                        for action, count in policy_dict.items()
                    }
                else:
                    normalized_policy = {}
                
                policy_targets.append(normalized_policy)
        
        return training_data, policy_targets
    
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
