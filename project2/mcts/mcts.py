from .node import Node
import math
import random

class MCTS():

    def __init__(self, gsm, nn_pred=None):
        self.gsm = gsm
        self.nn_pred = nn_pred
        self.num_simulations = 10
        self.c = 2

    def search(self, state):
        root = Node(state)

        for _ in range(self.num_simulations):
            self._run_simulation(root)

        action = self._best_action(root)
        value = self._get_value(root)
        policy = self._get_policy(root)

        return action, policy, value
    

    def _run_simulation(self, node):

        # 1. Selection
        while not self.gsm.is_terminal(node.state) and node.is_expanded():
            node = self._select(node)
        
        # 2. Expansion
        if not self.gsm.is_terminal(node.state):
            node = self._expand(node)

        # 3. Rollout
        value = self._rollout(node)

        # 4. Backprop
        self._backpropogation(node, value)


    def _select(self, node):
        best_score = -math.inf
        best_action = None

        for action in node.action_stats:
            stats = node.action_stats[action]
            if stats["N"] == 0:
                return node.children[action]
            
            Q = stats["Q"] 
            U = self.c * math.sqrt( math.log(node.visits) / stats["N"])
            score = Q + U

            if score > best_score:
                best_action = action
                best_score = score
    
        return node.children[best_action]

    def _expand(self, node):
        state = node.state
        actions = self.gsm.legal_actions(state)
        
        action = random.choice(actions)
        new_state = self.gsm.next_state(state, action)

        child = Node(new_state, parent=node, parent_action=action)
        node.add_child(action, child)
        
        return child

    def _rollout(self, node):
        state = node.state
        total_reward = 0

        while not self.gsm.is_terminal(state):
            actions = self.gsm.legal_actions(state)

            if not actions:
                return total_reward
            
            if self.nn_pred is not None:
                action = self.nn_pred(state) # actual input is sigma
            else:
                action = random.choice(actions)
            next_state = self.gsm.next_state(state, action)
            reward = self.gsm.reward(state, action, next_state)
            total_reward += reward
            state = next_state
        
        return total_reward


    def _backpropogation(self, node, value):
        while node is not None:
            if node.parent_action is not None:
                node.parent.update(node.parent_action, value)
            node = node.parent

    def _best_action(self, root):
        best_action = None
        best_visits = -1
        
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        
        return best_action
    
    def _get_policy(self, node):
        total_visits = sum(stats["N"] for stats in node.action_stats.values())
        if total_visits == 0:
            return {}
        
        policy = {}
        
        for action, stats in node.action_states.values():
                policy[action] = stats["N"] / total_visits
        
        return policy

    def _get_value(self, node):
        total_q = sum(stats["Q"] for stats in node.action_stats.values())
        total_n = sum(stats["N"] for stats in node.action_stats.values())
        return total_q / total_n if total_n > 0 else 0