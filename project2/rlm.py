"""
Reinforcement Learning Manager (RLM)

Handles high-level training operations:
- Run episodes with MCTS
- Collect data into episodes
- Train networks
- Manage training loop
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jax.numpy as jnp

from game.LineWorld import LineWorld
from mcts.mcts import MCTS
from nn.NNManager import NNManager
from buffer import EpisodeBuffer


class ReinforcementLearningManager:
    """Manages the overall training loop and episodes."""
    
    def __init__(self, game_state_manager, nn_manager, num_actions=2, trunk_name="trunk", value_name="value"):
        """Initialize RLM.
        
        Args:
            game_state_manager: GSM instance (e.g., LineWorld)
            nn_manager: NNM instance with networks
            num_actions: Number of possible actions
            trunk_name: Name of trunk network in NNM
            value_name: Name of value head network in NNM
        """
        self.gsm = game_state_manager
        self.nnm = nn_manager
        self.num_actions = num_actions
        
        # Get networks (allow custom names)
        try:
            self.trunk = nn_manager.get_net(trunk_name)
            self.value_head = nn_manager.get_net(value_name)
        except ValueError:
            # Fall back to first two networks if specific names don't exist
            available = list(nn_manager.models.keys())
            if len(available) >= 2:
                self.trunk = nn_manager.get_net(available[0])
                self.value_head = nn_manager.get_net(available[1])
            else:
                raise ValueError(f"Need at least 2 networks, found {len(available)}")
        
        # Episode buffer
        self.episode_buffer = EpisodeBuffer()
        
        # MCTS with NN
        self.nn_pred = self._make_nn_predictor()
        self.mcts = MCTS(self.gsm, num_actions=num_actions, nn_pred=self.nn_pred)
    
    def _make_nn_predictor(self):
        """Create a function that predicts value using the NN."""
        def predict(state):
            state_batch = jnp.array([[state]], dtype=jnp.float32)
            trunk_out = self.trunk(state_batch)
            value = float(self.value_head(trunk_out)[0, 0])
            return value, {}
        return predict
    
    def run_episode(self):
        """Run one episode of the game using MCTS for action selection.
        
        Returns:
            Episode dict with states, actions, rewards, policies, values
        """
        states = []
        actions_taken = []
        rewards = []
        policies = []
        values = []
        
        state = self.gsm.initial_state()
        total_reward = 0
        
        while not self.gsm.is_terminal(state):
            states.append(state)
            
            # MCTS action selection
            action, policy, value = self.mcts.search(state)
            actions_taken.append(action)
            policies.append(policy)
            values.append(value)
            
            # Take action
            next_state = self.gsm.next_state(state, action)
            reward = self.gsm.reward(state, action, next_state)
            rewards.append(reward)
            total_reward += reward
            
            state = next_state
        
        # Terminal state reached
        final_value = 1.0 if state == self.gsm.max_position else -1.0
        
        episode = {
            'states': states,
            'actions': actions_taken,
            'rewards': rewards,
            'policies': policies,
            'values': values,
            'final_value': final_value,
            'total_reward': total_reward
        }
        
        return episode
    
    def collect_episodes(self, num_episodes, use_mcts_nn=True):
        """Collect multiple episodes and store in buffer.
        
        Args:
            num_episodes: Number of episodes to collect
            use_mcts_nn: If True, use NN in MCTS; if False, use pure MCTS
        
        Returns:
            List of episodes
        """
        # Optionally disable NN for pure MCTS collection
        if not use_mcts_nn:
            self.mcts.nn_pred = None
        
        episodes = []
        for ep in range(num_episodes):
            episode = self.run_episode()
            episodes.append(episode)
            self.episode_buffer.add_episode(
                episode['states'],
                episode['actions'],
                episode['rewards'],
                episode['policies'],
                episode['values']
            )
            
            if (ep + 1) % max(1, num_episodes // 5) == 0:
                print(f"  Episode {ep+1}/{num_episodes}: "
                      f"reward={episode['total_reward']:+.1f}")
        
        # Re-enable NN for training
        if not use_mcts_nn:
            self.mcts.nn_pred = self.nn_pred
        
        return episodes
    
    def train_networks(self, num_epochs=20, learning_rate=0.01):
        """Train the networks using collected episodes.
        
        This delegates to NNManager.bptt_train() which handles BPTT.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
        """
        # Get training data from buffer
        training_data = self.episode_buffer.get_training_data()
        
        # Delegate to NNManager to handle BPTT
        self.nnm.bptt_train(
            trunk_name="trunk",
            value_name="value",
            training_data=training_data,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
    
    def training_loop(self, num_iterations=5, episodes_per_iter=10, training_epochs=20):
        """Main training loop.
        
        Args:
            num_iterations: Number of training iterations
            episodes_per_iter: Episodes to collect per iteration
            training_epochs: Epochs for network training per iteration
        """
        print(f"\n{'='*60}")
        print(f"REINFORCEMENT LEARNING LOOP")
        print(f"{'='*60}")
        
        for iteration in range(num_iterations):
            print(f"\n[Iteration {iteration+1}/{num_iterations}]")
            
            # Step 1: Collect episodes
            print(f"  Collecting {episodes_per_iter} episodes with MCTS+NN...")
            self.collect_episodes(episodes_per_iter, use_mcts_nn=True)
            
            # Step 2: Train networks
            print(f"  Training networks...")
            self.train_networks(num_epochs=training_epochs, learning_rate=0.01)
            
            # Step 3: Report
            print(f"  Buffer: {self.episode_buffer.size()} episodes total")
    
    def evaluate(self, num_games=20):
        """Evaluate current MCTS+NN against pure MCTS.
        
        Args:
            num_games: Number of games to play
        
        Returns:
            Dict with results
        """
        print(f"\nEvaluating ({num_games} games)...")
        
        # Pure MCTS test
        mcts_pure = MCTS(self.gsm, num_actions=self.num_actions, nn_pred=None)
        wins_pure = 0
        for i in range(num_games):
            state = self.gsm.initial_state()
            while not self.gsm.is_terminal(state):
                action, _, _ = mcts_pure.search(state)
                state = self.gsm.next_state(state, action)
            if state == self.gsm.max_position:
                wins_pure += 1
        
        # MCTS+NN test
        wins_nn = 0
        for i in range(num_games):
            state = self.gsm.initial_state()
            while not self.gsm.is_terminal(state):
                action, _, _ = self.mcts.search(state)
                state = self.gsm.next_state(state, action)
            if state == self.gsm.max_position:
                wins_nn += 1
        
        result = {
            'pure_mcts': wins_pure,
            'mcts_nn': wins_nn,
            'total': num_games,
            'pure_mcts_pct': 100 * wins_pure / num_games,
            'mcts_nn_pct': 100 * wins_nn / num_games
        }
        
        print(f"  Pure MCTS: {wins_pure}/{num_games} ({result['pure_mcts_pct']:.0f}%)")
        print(f"  MCTS+NN:   {wins_nn}/{num_games} ({result['mcts_nn_pct']:.0f}%)")
        
        return result


