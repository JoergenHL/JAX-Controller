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
    
    def train_repr_pred(self, states, value_targets, policy_targets,
                        num_epochs: int = 20, learning_rate: float = 0.01):
        """Train NNr and NNp jointly via backprop through the chained NNr → NNp pipeline.

        Key insight: NNr has no direct supervision targets. It only receives gradients
        that flow backward from NNp's loss — whatever NNr encodes must be useful for
        NNp to predict value and policy correctly.  This is the core BPTT mechanism
        in MuZero (without NNd for now).

        JAX pattern: jax.value_and_grad with argnums=(0, 1) differentiates the loss
        with respect to both parameter sets simultaneously in a single forward+backward
        pass, correctly chaining gradients through NNr → NNp.
        """
        nn_r = self.get_net("nnr")
        nn_p = self.get_net("nnp")

        states_batch = jnp.array(states).reshape(-1, 1)
        value_t  = jnp.array(value_targets, dtype=jnp.float32)
        policy_t = jnp.array(policy_targets, dtype=jnp.float32)

        gdef_r, params_r = nnx.split(nn_r)
        gdef_p, params_p = nnx.split(nn_p)

        def loss_fn(params_r, params_p):
            abstract      = nnx.merge(gdef_r, params_r)(states_batch)   # NNr forward
            output        = nnx.merge(gdef_p, params_p)(abstract)        # NNp forward
            value_pred    = output[:, 0]
            policy_logits = output[:, 1:]
            value_loss  = jnp.mean((value_pred - value_t) ** 2)
            log_probs   = jax.nn.log_softmax(policy_logits, axis=-1)
            policy_loss = -jnp.mean(jnp.sum(policy_t * log_probs, axis=-1))
            return value_loss + policy_loss, (value_loss, policy_loss)

        print(f"  Training NNr+NNp jointly for {num_epochs} epochs...")

        history = []
        for epoch in range(num_epochs):
            (loss, (v_loss, p_loss)), (grads_r, grads_p) = jax.value_and_grad(
                loss_fn, argnums=(0, 1), has_aux=True
            )(params_r, params_p)
            params_r = jax.tree_util.tree_map(
                lambda w, g: w - learning_rate * g, params_r, grads_r
            )
            params_p = jax.tree_util.tree_map(
                lambda w, g: w - learning_rate * g, params_p, grads_p
            )
            history.append((float(loss), float(v_loss), float(p_loss)))

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{num_epochs}: "
                      f"loss={float(loss):.4f}  "
                      f"(value={float(v_loss):.4f}, policy={float(p_loss):.4f})")

        nnx.update(nn_r, params_r)
        nnx.update(nn_p, params_p)
        return history

    def train(self, net_name: str, states, value_targets, policy_targets,
              num_epochs: int = 20, learning_rate: float = 0.01):
        """Train network via full backprop through all layers simultaneously."""
        net = self.get_net(net_name)

        states_batch = jnp.array(states).reshape(-1, 1)
        value_t = jnp.array(value_targets, dtype=jnp.float32)
        policy_t = jnp.array(policy_targets, dtype=jnp.float32)

        # nnx.split separates the model into:
        #   graphdef — static structure (layer shapes, connectivity)
        #   params   — all trainable weights and biases as a JAX pytree
        # Differentiating through 'params' gives correct gradients for every
        # layer simultaneously (true backprop), unlike updating layers one-by-one.
        graphdef, params = nnx.split(net)

        def loss_fn(params):
            model = nnx.merge(graphdef, params)
            output = model(states_batch)       # [N, 1 + num_actions]
            value_pred    = output[:, 0]
            policy_logits = output[:, 1:]

            value_loss  = jnp.mean((value_pred - value_t) ** 2)
            log_probs   = jax.nn.log_softmax(policy_logits, axis=-1)
            policy_loss = -jnp.mean(jnp.sum(policy_t * log_probs, axis=-1))
            total_loss  = value_loss + policy_loss
            return total_loss, (value_loss, policy_loss)

        print(f"  Training {net_name} for {num_epochs} epochs...")

        history = []  # (total, value, policy) per epoch
        for epoch in range(num_epochs):
            (loss, (v_loss, p_loss)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(params)
            # SGD: w ← w - lr * ∂L/∂w  (applied to every parameter at once)
            params = jax.tree_util.tree_map(
                lambda w, g: w - learning_rate * g, params, grads
            )
            history.append((float(loss), float(v_loss), float(p_loss)))

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{num_epochs}: "
                      f"loss={float(loss):.4f}  "
                      f"(value={float(v_loss):.4f}, policy={float(p_loss):.4f})")

        # Write the updated parameters back into the model object in-place.
        nnx.update(net, params)
        return history



