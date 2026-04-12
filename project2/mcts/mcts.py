from .node import Node
import math
import random
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


# Module-level JIT-compiled forward pass for any NNX model.
# nnx.jit (not jax.jit) is required here: it extracts the NNX module's current
# parameter state before each call, so updated weights after training are always
# used. jax.jit would capture parameters at trace time and ignore later updates.
_net_fwd = nnx.jit(lambda model, x: model(x))


class MCTS:
    """u-MCTS: Monte Carlo Tree Search operating entirely in abstract (latent) space.

    Stage 4B change from Stage 3/4A:
      Before: MCTS expanded nodes by calling gsm.next_state() — the real game simulator.
      Now:    MCTS expands nodes by calling NNd directly — the learned dynamics model.
              The real game is never called during search.

    Search flow:
        real_state ──NNr──> σ_root
        for each simulation:
            select leaf L via PUCT
            expand: all children via one batched NNd call
            evaluate: leaf value = NNp(σ_leaf).value
            backpropagate value to root
    """

    def __init__(self, nn_r, nn_d, nn_p, action_space, use_puct=True,
                 dir_alpha=0.3, dir_epsilon=0.25):
        """
        Args:
            nn_r:         Representation network (NNr): real_state → σ
            nn_d:         Dynamics network (NNd): (σ, action) → (next_σ, reward)
            nn_p:         Prediction network (NNp): σ → (value, policy_logits)
            action_space: Ordered list of all legal actions (e.g. ["LEFT", "RIGHT"])
            use_puct:     If True, use PUCT selection; otherwise standard UCB
            dir_alpha:    Dirichlet concentration parameter for root exploration noise
            dir_epsilon:  Weight given to Dirichlet noise vs network prior at root
        """
        self.nn_r         = nn_r
        self.nn_d         = nn_d
        self.nn_p         = nn_p
        self.action_space = action_space
        self.use_puct     = use_puct
        self.dir_alpha    = dir_alpha
        self.dir_epsilon  = dir_epsilon
        self.num_simulations = 10
        self.c    = 2
        self.d_max = 10

    def search(self, real_state):
        """Run u-MCTS from a real game state and return best action.

        Encodes the real state to abstract space once at the root; all
        subsequent expansion and evaluation happens in abstract space.

        Returns:
            action: Best action according to visit counts
            policy: Dict {action: visit_count} — training target for NNp
            value:  Root value estimate
        """
        sigma = _net_fwd(self.nn_r, jnp.array([[real_state]], dtype=jnp.float32))
        root  = Node(sigma)

        for _ in range(self.num_simulations):
            self._run_simulation(root)

        return self._best_action(root), self._get_policy(root), self._get_value(root)

    def _run_simulation(self, root):
        """One simulation: selection → expansion → evaluation → backprop."""
        node, depth = root, 0

        # Selection: follow tree policy until an unexpanded node or max depth
        while node.is_expanded() and depth < self.d_max:
            node = self._select(node)
            depth += 1

        # Expansion: add all children via NNd
        if depth < self.d_max:
            self._expand(node)

        # Evaluation: estimate leaf value with NNp
        value = self._evaluate(node)

        self._backpropogation(node, value)

    def _select(self, node):
        """PUCT action selection.

        Visits unvisited children first (random order to avoid dict-ordering bias),
        then selects by Q + c * P * sqrt(N_parent) / (1 + N_child).
        """
        unvisited = [a for a, s in node.action_stats.items() if s["N"] == 0]
        if unvisited:
            return node.children[random.choice(unvisited)]

        best_score  = -math.inf
        best_action = None

        for action, stats in node.action_stats.items():
            Q = stats["Q"]
            p = stats["policy_prior"]
            N = stats["N"]
            U = self.c * p * math.sqrt(node.visits) / (1 + N)
            if Q + U > best_score:
                best_score  = Q + U
                best_action = action

        return node.children[best_action]

    def _expand(self, node):
        """Add one child per action using NNd; store NNp policy priors.

        All actions are always legal in abstract space — there is no terminal
        check here. NNd predicts where each action leads in latent space.

        Key optimisation: all NNd transitions are computed in ONE batched
        forward pass rather than one call per action. The input batch is:
            [[σ, onehot(a0)],
             [σ, onehot(a1)], ...]   shape [num_actions, abstract_dim + num_actions]
        A single _net_fwd call returns all next-σ and predicted rewards at once.
        """
        sigma        = node.state                        # [1, abstract_dim]
        abstract_dim = sigma.shape[1]
        num_actions  = len(self.action_space)

        # Policy priors — one NNp call (JIT compiled by _net_fwd)
        nnp_out      = _net_fwd(self.nn_p, sigma)[0]    # [value, logit_L, logit_R]
        policy_probs = jax.nn.softmax(nnp_out[1:])

        # At the root, mix in Dirichlet noise (AlphaZero convention).
        # Without noise, a slight initial bias in NNp feeds back through MCTS visit
        # counts → policy targets → training → stronger bias (policy collapse).
        # Dirichlet noise breaks this loop by forcing the root to explore all actions.
        if node.parent is None:
            noise        = jnp.array(np.random.dirichlet([self.dir_alpha] * num_actions))
            policy_probs = (1 - self.dir_epsilon) * policy_probs + self.dir_epsilon * noise

        # Batch all NNd calls: tile σ and append each action's one-hot
        # Result: one XLA kernel dispatch instead of num_actions separate calls
        nnd_input  = jnp.concatenate([
            jnp.tile(sigma, (num_actions, 1)),           # [num_actions, abstract_dim]
            jnp.eye(num_actions),                        # [num_actions, num_actions]
        ], axis=1)                                       # [num_actions, abstract_dim + num_actions]
        nnd_output = _net_fwd(self.nn_d, nnd_input)     # [num_actions, abstract_dim + 1]

        for action_idx, action in enumerate(self.action_space):
            next_sigma = nnd_output[action_idx : action_idx + 1, :abstract_dim]
            child = Node(next_sigma, parent=node, parent_action=action)
            node.add_child(action, child)
            node.action_stats[action]["policy_prior"] = float(policy_probs[action_idx])

    def _evaluate(self, node):
        """Estimate the value of an abstract leaf state using NNp."""
        return float(_net_fwd(self.nn_p, node.state)[0, 0])

    def _backpropogation(self, node, value):
        """Back up value through all ancestors to the root."""
        while node is not None:
            if node.parent_action is not None:
                node.parent.update(node.parent_action, value)
            node = node.parent

    def _best_action(self, root):
        """Return the action with the highest visit count at the root.

        Ties are broken randomly to avoid systematic bias from dict insertion order.
        With uniform values and policy, PUCT equalizes visit counts; without random
        tie-breaking the first action in insertion order would always win.
        """
        max_n = max(stats["N"] for stats in root.action_stats.values())
        best  = [a for a, s in root.action_stats.items() if s["N"] == max_n]
        return random.choice(best)

    def _get_policy(self, node):
        """Return visit-count distribution over actions (training target for NNp)."""
        return {action: stats["N"] for action, stats in node.action_stats.items()}

    def _get_value(self, node):
        """Root value: total accumulated value / total visits."""
        total_w = sum(stats["W"] for stats in node.action_stats.values())
        total_n = sum(stats["N"] for stats in node.action_stats.values())
        return total_w / total_n if total_n > 0 else 0
