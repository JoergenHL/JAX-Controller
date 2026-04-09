from .node import Node
import math
import random

class MCTS():

    def __init__(self, gsm, num_actions, nn_pred=None):
        """Initialize MCTS.
        
        Args:
            gsm: Game state manager
            num_actions: Number of legal actions
            nn_pred: Optional function(state) -> (value, policy_dict)
                    Returns value estimate and policy probabilities over actions
        """
        self.gsm = gsm
        self.nn_pred = nn_pred
        self.num_actions = num_actions
        self.num_simulations = 10
        self.c = 2

    def search(self, state, debug=False):
        """Run MCTS from a state and return best action with policy and value.
        
        Returns:
            (action, policy, value) where:
            - action: Best action to take
            - policy: Dict of visit counts {action: count}
            - value: Root value estimate
        """
        root = Node(state)

        for _ in range(self.num_simulations):
            self._run_simulation(root)

        action = self._best_action(root)
        value = self._get_value(root)
        policy = self._get_policy(root)
        
        if debug:
            print(f"\n  DEBUG state={state}, action_stats={root.action_stats}")
            print(f"  Q values at root:")
            for action, stats in root.action_stats.items():
                q_val = stats["Q"] if stats["N"] > 0 else "N/A"
                print(f"    action {action}: Q={q_val:.4f}, N={stats['N']}, W={stats['W']:.4f}")

        return action, policy, value
    

    def _run_simulation(self, node):
        """Run one simulation: selection → expansion → evaluation → backprop."""
        # 1. Selection
        while not self.gsm.is_terminal(node.state) and node.is_expanded():
            node = self._select(node)
        
        # 2. Expansion
        if not self.gsm.is_terminal(node.state):
            node = self._expand(node)

        # 3. Evaluation: get value estimate from NN or random rollout
        value = self._evaluate(node)

        # 4. Backprop
        self._backpropogation(node, value)
    
    def _debug_tree(self, node, depth=0):
        """Print the Q values for a node's children."""
        indent = "  " * depth
        for action, stats in node.action_stats.items():
            q_val = stats["Q"] if stats["N"] > 0 else "N/A"
            print(f"{indent}  action {action}: Q={q_val:.3f}, N={stats['N']}, W={stats['W']:.3f}")
        if depth < 2:
            for action, child in node.children.items():
                print(f"{indent}child {action} (state={child.state}):")
                self._debug_tree(child, depth+1)


    def _select(self, node):
        """UCB1 action selection."""
        best_score = -math.inf
        best_action = None

        for action in node.action_stats:
            stats = node.action_stats[action]
            if stats["N"] == 0:
                return node.children[action]
            
            Q = stats["Q"] 
            U = self.c * math.sqrt(math.log(node.visits) / stats["N"])
            score = Q + U

            if score > best_score:
                best_action = action
                best_score = score
    
        return node.children[best_action]

    def _expand(self, node):
        """Expand node by adding ALL legal action children."""
        state = node.state
        actions = self.gsm.legal_actions(state)
        
        # Initialize all legal actions with children and stats
        for action in actions:
            new_state = self.gsm.next_state(state, action)
            child = Node(new_state, parent=node, parent_action=action)
            node.add_child(action, child)
        
        # Return one of the children for evaluation
        return node.children[random.choice(actions)]

    def _evaluate(self, node):
        """Evaluate a node using NN value or random rollout.
        
        If NN available: use predicted value (fast)
        If NN unavailable: use random rollout (slow but model-free)
        """
        state = node.state
        
        if self.nn_pred is not None:
            # Use NN to estimate value
            value, _ = self.nn_pred(state)
            return value
        else:
            # Random rollout
            return self._random_rollout(state)

    def _random_rollout(self, state):
        """Random playout to terminal state."""
        total_reward = 0
        
        while not self.gsm.is_terminal(state):
            actions = self.gsm.legal_actions(state)
            if not actions:
                break
            
            action = random.choice(actions)
            next_state = self.gsm.next_state(state, action)
            reward = self.gsm.reward(state, action, next_state)
            total_reward += reward
            state = next_state
        
        return total_reward

    def _backpropogation(self, node, value):
        """Backup value up the tree."""
        while node is not None:
            if node.parent_action is not None:
                node.parent.update(node.parent_action, value)
            node = node.parent

    def _best_action(self, root):
        """Select action with most visits (edge visit counts)."""
        best_action = None
        best_visits = -1
        
        for action, stats in root.action_stats.items():
            if stats["N"] > best_visits:
                best_visits = stats["N"]
                best_action = action
        
        return best_action
    
    def _get_policy(self, node):
        """Get visit count distribution over actions."""
        total_visits = sum(stats["N"] for stats in node.action_stats.values())
        if total_visits == 0:
            return {}
        
        policy = {}
        
        for action, stats in node.action_stats.items():
            policy[action] = stats["N"] / total_visits
        
        return policy

    def _get_value(self, node):
        """Get root value: total accumulated reward / total visits."""
        total_w = sum(stats["W"] for stats in node.action_stats.values())
        total_n = sum(stats["N"] for stats in node.action_stats.values())
        return total_w / total_n if total_n > 0 else 0