from .node import Node
import math
import random
import jax.nn


class MCTS():

    def __init__(self, gsm, num_actions, nn_pred=None, use_puct=False):
        """Initialize MCTS.
        
        Args:
            gsm: Game state manager
            num_actions: Number of legal actions
            nn_pred: Optional function(state) -> (value, policy_logits)
                    Returns value estimate and policy logits over actions
            use_puct: If True, use PUCT (policy-guided UCB) instead of UCB
        """
        self.gsm = gsm
        self.nn_pred = nn_pred
        self.num_actions = num_actions
        self.num_simulations = 10
        self.c = 2
        self.use_puct = use_puct

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
        
        # 2. Expansion — expand the leaf in-place; keep node pointing at the leaf
        if not self.gsm.is_terminal(node.state):
            self._expand(node)

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
        """Action selection: PUCT (policy-guided UCB).
        
        Uses policy logits to bias exploration toward promising actions,
        while still using Q-values to guide the search.
        """
        # Pick randomly among unvisited actions to avoid dict-order bias.
        unvisited = [a for a, s in node.action_stats.items() if s["N"] == 0]
        if unvisited:
            return node.children[random.choice(unvisited)]

        best_score = -math.inf
        best_action = None

        for action, stats in node.action_stats.items():
            Q = stats["Q"]
            
            if "policy_prior" in stats:
                # PUCT: Q(s,a) + c * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
                p = stats["policy_prior"]
                N = stats["N"]
                U = self.c * p * math.sqrt(node.visits) / (1 + N)
            else:
                # Fallback UCB if policy not available
                U = self.c * math.sqrt(math.log(node.visits) / stats["N"])
            
            score = Q + U

            if score > best_score:
                best_action = action
                best_score = score
    
        return node.children[best_action]

    def _expand(self, node):
        """Expand node and extract policy priors to guide future selection.
        
        Get policy logits from network to bias action exploration (PUCT).
        Value will be backpropagated from leaf evaluation.
        """
        state = node.state
        actions = self.gsm.legal_actions(state)
        
        # Get policy priors from NN to guide exploration
        policy_priors = None
        if self.nn_pred is not None:
            try:
                _, policy_logits = self.nn_pred(state)
                # Convert logits to probabilities via softmax
                policy_probs = jax.nn.softmax(policy_logits)
                policy_priors = [float(policy_probs[i]) for i in range(len(policy_probs))]
            except Exception:
                policy_priors = None
        
        # Initialize all legal actions with children and stats
        for action_idx, action in enumerate(actions):
            new_state = self.gsm.next_state(state, action)
            child = Node(new_state, parent=node, parent_action=action)
            node.add_child(action, child)
            
            # Store policy prior to bias selection (used in PUCT)
            if policy_priors is not None and action_idx < len(policy_priors):
                node.action_stats[action]["policy_prior"] = policy_priors[action_idx]
        
        # Expansion complete — caller evaluates the leaf itself, not a child

    def _evaluate(self, node):
        """Evaluate a node using NN value or random rollout.
        
        If NN available: use predicted value (fast)
        If NN unavailable: use random rollout (slow but model-free)
        
        Returns:
            value: Scalar value estimate
        """
        state = node.state

        # Terminal states have a known, exact value from the game itself.
        # The network has never been trained on terminal states and would return
        # a meaningless ~0; we must use the actual reward here so MCTS
        # backpropagates real signal (+1 / -1) rather than noise.
        if self.gsm.is_terminal(state):
            if node.parent is not None and node.parent_action is not None:
                return self.gsm.reward(node.parent.state, node.parent_action, state)
            return 0.0

        if self.nn_pred is not None:
            try:
                # NN returns (value, policy_logits)
                result = self.nn_pred(state)
                if isinstance(result, tuple):
                    value, _ = result
                else:
                    value = result
                return float(value)
            except Exception:
                return self._random_rollout(state)
        else:
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
        """Get visit count distribution over actions (for policy training target)."""
        policy = {}
        for action, stats in node.action_stats.items():
            policy[action] = stats["N"]
        return policy

    def _get_value(self, node):
        """Get root value: total accumulated reward / total visits."""
        total_w = sum(stats["W"] for stats in node.action_stats.values())
        total_n = sum(stats["N"] for stats in node.action_stats.values())
        return total_w / total_n if total_n > 0 else 0