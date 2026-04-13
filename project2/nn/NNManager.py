from flax import nnx
import jax
import jax.numpy as jnp

from nn.nn import MLP


class NNManager:
    """Owns and trains neural networks."""
    
    def __init__(self):
        self.models = {}
    
    def create_net(self, name: str, dims: list[int]):
        """Create network with given dimensions."""
        model = MLP(dims, rngs=nnx.Rngs(1))
        self.models[name] = model
        return model
    
    def get_net(self, name: str):
        """Get network by name."""
        if name not in self.models:
            raise ValueError(f"Network '{name}' not found")
        return self.models[name]
    
    def train_bptt(self, minibatches, abstract_dim, num_actions,
                   num_epochs: int = 20, learning_rate: float = 0.01):
        """Full BPTT through the NNr → NNd^w → NNp composite network.

        This is the core MuZero training procedure (PDF: DO_BPTT_TRAINING).

        For each minibatch sample:
          1. Encode the real starting state: σ_k = NNr(s_k)
          2. Unroll NNd w times with the real actions taken in the episode
          3. At each step, call NNp(σ_{k+i}) → (v̂, π̂) and compare to targets
          4. Loss = value MSE + policy CE + reward MSE across all w steps
          5. jax.value_and_grad with argnums=(0,1,2) differentiates through the
             entire unrolled graph — gradients flow back through all three networks

        Speed: instead of a Python for-loop over ~700 windows per epoch (each
        triggering a separate JIT dispatch), all windows are stacked into JAX arrays
        and processed in one jax.vmap call per epoch. This compiles to a single XLA
        kernel that executes the entire batch in parallel, eliminating per-sample
        Python overhead.
        """
        nn_r = self.get_net("nnr")
        nn_d = self.get_net("nnd")
        nn_p = self.get_net("nnp")

        gdef_r, params_r = nnx.split(nn_r)
        gdef_d, params_d = nnx.split(nn_d)
        gdef_p, params_p = nnx.split(nn_p)

        roll_ahead = len(minibatches[0]['action_indices'])

        # Stack the entire minibatch into batched JAX arrays once — outside the
        # epoch loop — so array construction cost is paid only once per training call.
        batch_states   = jnp.array([s['state']          for s in minibatches], dtype=jnp.float32)
        batch_actions  = jnp.array([s['action_indices']  for s in minibatches], dtype=jnp.int32)
        batch_v_t      = jnp.array([s['value_targets']   for s in minibatches], dtype=jnp.float32)
        batch_p_t      = jnp.array([s['policy_targets']  for s in minibatches], dtype=jnp.float32)
        batch_r_t      = jnp.array([s['reward_targets']  for s in minibatches], dtype=jnp.float32)

        def loss_for_one(params_r, params_d, params_p,
                         state, action_indices, v_targets, p_targets, r_targets):
            """Unrolled BPTT loss for one episode window.

            Called via jax.vmap — params are shared (not batched), data is per-sample.
            Gradients flow: loss ← NNp ← NNd ← NNd ← NNd ← NNr (one pass).
            """
            sigma = nnx.merge(gdef_r, params_r)(jnp.atleast_2d(state))  # [1, state_dim] → [1, abstract_dim]

            total_v = jnp.array(0.0)
            total_p = jnp.array(0.0)
            total_r = jnp.array(0.0)

            for step in range(roll_ahead):
                output = nnx.merge(gdef_p, params_p)(sigma)[0]  # [value, logit_L, logit_R]
                v_hat, pi_hat = output[0], output[1:]

                total_v = total_v + (v_hat - v_targets[step]) ** 2
                total_p = total_p - jnp.sum(p_targets[step] * jax.nn.log_softmax(pi_hat))

                action_onehot = jnp.zeros(num_actions).at[action_indices[step]].set(1.0)
                nnd_input  = jnp.concatenate([sigma[0], action_onehot])[None, :]
                nnd_output = nnx.merge(gdef_d, params_d)(nnd_input)[0]  # [abstract_dim + 1]
                sigma      = nnd_output[:abstract_dim][None, :]          # [1, abstract_dim]

                total_r = total_r + (nnd_output[-1] - r_targets[step]) ** 2

            return total_v / roll_ahead, total_p / roll_ahead, total_r / roll_ahead

        def batch_loss(params_r, params_d, params_p):
            """Mean loss over all windows, computed via vmap.

            jax.vmap vectorises loss_for_one across the batch dimension:
              - params (argnums 0-2): in_axes=None — shared across all samples
              - data (argnums 3-7):   in_axes=0    — one slice per sample
            The Python for-loop inside loss_for_one is unrolled at trace time
            (roll_ahead is a static integer), then the entire unrolled graph is
            vectorised. One JIT compilation covers all batch sizes with the same shape.
            """
            v_l, p_l, r_l = jax.vmap(
                loss_for_one, in_axes=(None, None, None, 0, 0, 0, 0, 0)
            )(params_r, params_d, params_p,
              batch_states, batch_actions, batch_v_t, batch_p_t, batch_r_t)
            v_mean = jnp.mean(v_l)
            p_mean = jnp.mean(p_l)
            r_mean = jnp.mean(r_l)
            return v_mean + p_mean + r_mean, (v_mean, p_mean, r_mean)

        # Compile once: the entire batch forward + backward pass becomes one XLA kernel.
        grad_fn = jax.jit(
            jax.value_and_grad(batch_loss, argnums=(0, 1, 2), has_aux=True)
        )

        print(f"  Training NNr+NNd+NNp via BPTT for {num_epochs} epochs "
              f"({len(minibatches)} windows, roll_ahead={roll_ahead})...")

        def _clip_grads(grads, max_norm=1.0):
            """Clip gradient tree by global L2 norm to prevent explosion."""
            leaves = jax.tree_util.tree_leaves(grads)
            global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
            scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))
            return jax.tree_util.tree_map(lambda g: g * scale, grads)

        history = []
        for epoch in range(num_epochs):
            (_, (v_loss, p_loss, r_loss)), (gr, gd, gp) = grad_fn(
                params_r, params_d, params_p
            )
            gr, gd, gp = _clip_grads(gr), _clip_grads(gd), _clip_grads(gp)
            params_r = jax.tree_util.tree_map(lambda w, g: w - learning_rate * g, params_r, gr)
            params_d = jax.tree_util.tree_map(lambda w, g: w - learning_rate * g, params_d, gd)
            params_p = jax.tree_util.tree_map(lambda w, g: w - learning_rate * g, params_p, gp)

            total = float(v_loss) + float(p_loss) + float(r_loss)
            history.append((total, float(v_loss), float(p_loss), float(r_loss)))

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{num_epochs}: "
                      f"loss={total:.4f}  "
                      f"(value={float(v_loss):.4f}, "
                      f"policy={float(p_loss):.4f}, "
                      f"reward={float(r_loss):.4f})")

        nnx.update(nn_r, params_r)
        nnx.update(nn_d, params_d)
        nnx.update(nn_p, params_p)
        return history



