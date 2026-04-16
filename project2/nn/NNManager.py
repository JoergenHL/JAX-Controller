from flax import nnx
import jax
import jax.numpy as jnp
import optax

from nn.nn import MLP


class NNManager:
    """Owns and trains neural networks."""
    
    def __init__(self):
        self.models = {}
        # Persistent optimiser state across train_bptt calls (main process only —
        # workers never train, so this is never pickled into subprocesses).
        # Initialised lazily on the first train_bptt call, then reused every
        # iteration so Adam momentum / second-moment estimates accumulate the
        # consistent-across-iterations signal instead of being reset each time.
        self.optimizer  = None
        self.opt_state  = {}   # {net_name: opt_state}
        self._rng_key   = None
    
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
                   num_updates: int = 500, minibatch_size: int = 128,
                   learning_rate: float = 0.01):
        """MuZero-style minibatched BPTT through NNr → NNd^w → NNp.

        Each gradient step samples `minibatch_size` random (episode, step)
        windows from the full buffer and does one Adam update. Over `num_updates`
        steps this is equivalent to the pseudocode's `for m in range(mbs)` loop
        with many iterations.

        Per-sample loss (via jax.vmap over the sampled indices):
          1. σ = NNr(s_k)
          2. Unroll NNd w steps with the real actions
          3. NNp(σ) at each step → compare to (value, policy) targets
          4. Loss = value MSE + policy CE + reward MSE, mean over unroll
          5. jax.value_and_grad with argnums=(0,1,2) differentiates through
             the entire unrolled graph

        Optimiser state is persistent across train_bptt calls (stored on self),
        so Adam momentum tracks the direction of travel as the buffer rotates
        instead of restarting from zero each iteration.
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

        # Loss weights: value is down-weighted following the MuZero paper (0.25)
        # to prevent it from being crowded out by the policy cross-entropy loss,
        # which tends to dominate on an equal-weight sum early in training.
        import config as _config
        lw = _config.nn.get("loss_weights", {"value": 0.25, "policy": 1.0, "reward": 1.0})
        w_v, w_p, w_r = lw["value"], lw["policy"], lw["reward"]

        def batch_loss(params_r, params_d, params_p,
                       mb_states, mb_actions, mb_v_t, mb_p_t, mb_r_t):
            """Mean loss over the sampled minibatch, computed via vmap.

            Params are shared across samples; data has a leading batch dim.
            The Python for-loop inside loss_for_one (over roll_ahead) is
            unrolled at trace time; the result is vmapped across the minibatch.
            """
            v_l, p_l, r_l = jax.vmap(
                loss_for_one, in_axes=(None, None, None, 0, 0, 0, 0, 0)
            )(params_r, params_d, params_p,
              mb_states, mb_actions, mb_v_t, mb_p_t, mb_r_t)
            v_mean = jnp.mean(v_l)
            p_mean = jnp.mean(p_l)
            r_mean = jnp.mean(r_l)
            return w_v * v_mean + w_p * p_mean + w_r * r_mean, (v_mean, p_mean, r_mean)

        # Persistent Adam across train_bptt calls (see __init__). Lazy init on
        # first call — at that point the network params exist and shapes are known.
        if self.optimizer is None:
            self.optimizer = optax.adam(learning_rate)
            self.opt_state["nnr"] = self.optimizer.init(params_r)
            self.opt_state["nnd"] = self.optimizer.init(params_d)
            self.opt_state["nnp"] = self.optimizer.init(params_p)
            self._rng_key = jax.random.PRNGKey(0)

        opt_state_r = self.opt_state["nnr"]
        opt_state_d = self.opt_state["nnd"]
        opt_state_p = self.opt_state["nnp"]

        def _clip_grads(grads, max_norm=1.0):
            """Clip gradient tree by global L2 norm to prevent explosion."""
            leaves = jax.tree_util.tree_leaves(grads)
            global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
            scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))
            return jax.tree_util.tree_map(lambda g: g * scale, grads)

        optimizer = self.optimizer

        # One fused JIT'd update: sample-indexing, forward, backward, Adam step.
        # The minibatch indices are passed in (not sampled inside) so the JIT
        # cache key stays stable across calls.
        @jax.jit
        def update_step(params_r, params_d, params_p,
                        opt_state_r, opt_state_d, opt_state_p, idx):
            mb_s = batch_states[idx]
            mb_a = batch_actions[idx]
            mb_v = batch_v_t[idx]
            mb_p = batch_p_t[idx]
            mb_r = batch_r_t[idx]
            (_, (v_loss, p_loss, r_loss)), (gr, gd, gp) = jax.value_and_grad(
                batch_loss, argnums=(0, 1, 2), has_aux=True
            )(params_r, params_d, params_p, mb_s, mb_a, mb_v, mb_p, mb_r)
            gr, gd, gp = _clip_grads(gr), _clip_grads(gd), _clip_grads(gp)
            upd_r, opt_state_r = optimizer.update(gr, opt_state_r, params_r)
            upd_d, opt_state_d = optimizer.update(gd, opt_state_d, params_d)
            upd_p, opt_state_p = optimizer.update(gp, opt_state_p, params_p)
            params_r = optax.apply_updates(params_r, upd_r)
            params_d = optax.apply_updates(params_d, upd_d)
            params_p = optax.apply_updates(params_p, upd_p)
            return (params_r, params_d, params_p,
                    opt_state_r, opt_state_d, opt_state_p,
                    v_loss, p_loss, r_loss)

        N = len(minibatches)
        mbs = min(minibatch_size, N)
        print(f"  Training NNr+NNd+NNp via BPTT: {num_updates} updates "
              f"(mbs={mbs}, buffer={N} windows, roll_ahead={roll_ahead})...")

        history = []
        for step in range(num_updates):
            self._rng_key, sub = jax.random.split(self._rng_key)
            idx = jax.random.randint(sub, (mbs,), 0, N)
            (params_r, params_d, params_p,
             opt_state_r, opt_state_d, opt_state_p,
             v_loss, p_loss, r_loss) = update_step(
                params_r, params_d, params_p,
                opt_state_r, opt_state_d, opt_state_p, idx,
            )
            total = float(v_loss) + float(p_loss) + float(r_loss)
            history.append((total, float(v_loss), float(p_loss), float(r_loss)))

            if (step + 1) % max(1, num_updates // 10) == 0 or step == 0:
                print(f"    Update {step+1}/{num_updates}: "
                      f"loss={total:.4f}  "
                      f"(value={float(v_loss):.4f}, "
                      f"policy={float(p_loss):.4f}, "
                      f"reward={float(r_loss):.4f})")

        # Persist optimiser state for the next iteration.
        self.opt_state["nnr"] = opt_state_r
        self.opt_state["nnd"] = opt_state_d
        self.opt_state["nnp"] = opt_state_p

        nnx.update(nn_r, params_r)
        nnx.update(nn_d, params_d)
        nnx.update(nn_p, params_p)
        return history

    # ── Weight serialisation ───────────────────────────────────────────────────

    def get_layer_weights(self) -> dict:
        """Extract all network weights as plain numpy arrays.

        Returns {name: [(w_np, b_np), ...]} — one list of (weight, bias) pairs
        per network, ordered by layer index. Used to send a frozen weight
        snapshot to worker processes without passing JAX objects.
        """
        import numpy as np
        return {
            name: [
                (np.asarray(layer.w.value), np.asarray(layer.b.value))
                for layer in model.layers
            ]
            for name, model in self.models.items()
        }

    def set_layer_weights(self, weights: dict):
        """Load numpy weight arrays into existing models.

        Args:
            weights: same structure as get_layer_weights() returns.
        """
        for name, layer_weights in weights.items():
            model = self.models[name]
            for layer, (w, b) in zip(model.layers, layer_weights):
                layer.w.value = jnp.array(w)
                layer.b.value = jnp.array(b)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Persist all network weights to a pickle file.

        Uses nnx.split to separate the static graph structure (GraphDef) from
        the trainable parameters (State). Both are required by nnx.merge on load.
        All networks in the manager are saved under their names as keys.
        """
        import pickle
        data = {}
        for name, model in self.models.items():
            gdef, state = nnx.split(model)
            data[name] = (gdef, state)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Model saved → {path}")

    def load(self, path: str):
        """Restore network weights from a pickle file.

        Reconstructs each model with nnx.merge and replaces the entry in
        self.models. Call this on an empty NNManager — no need to pre-create
        networks with create_net(); the GraphDef carries the architecture.
        """
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        for name, (gdef, state) in data.items():
            self.models[name] = nnx.merge(gdef, state)



