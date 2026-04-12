import jax.numpy as jnp


class ASM:
    """Abstract State Manager — bridges real game states and abstract (latent) states.

    In MuZero, all planning happens in abstract space. The ASM is the only place
    that knows how to move between the two worlds:

        real state  ──NNr──>  abstract state σ  ──NNp──>  (value, policy)
                                                  ──NNd──>  (next σ, reward)

    This class holds no network weights itself — it receives the networks as
    arguments so the NNManager remains the single owner of all parameters.
    """

    def map_abstract_state(self, state, nn_r):
        """Map a real game state to an abstract state vector via NNr.

        Args:
            state:  scalar real game state (e.g. position in LineWorld)
            nn_r:   the representation network (NNr)

        Returns:
            abstract_state: jnp array of shape [1, abstract_dim]
        """
        state_array = jnp.array([[state]], dtype=jnp.float32)
        return nn_r(state_array)   # shape: [1, abstract_dim]

    def predict(self, abstract_state, nn_p):
        """Predict value and policy from an abstract state via NNp.

        Args:
            abstract_state: jnp array of shape [1, abstract_dim]
            nn_p:           the prediction network (NNp)

        Returns:
            value:          scalar float — estimated return from this state
            policy_logits:  raw (pre-softmax) action preferences
        """
        output = nn_p(abstract_state)[0]   # shape: [1 + num_actions]
        value = float(output[0])
        policy_logits = output[1:]
        return value, policy_logits

    def next_abstract_state(self, abstract_state, action, nn_d, action_space):
        """Predict next abstract state and immediate reward via NNd.

        This is the learned world model: given a state in abstract space and
        an action, predict where the world goes next (still in abstract space)
        and what reward will be received. NNd never sees real game states.

        Input to NNd: [abstract_state (abstract_dim,) ++ action_onehot (num_actions,)]
        Output of NNd: [next_abstract_state (abstract_dim,), predicted_reward (1,)]

        Args:
            abstract_state: jnp array of shape [1, abstract_dim]
            action:         action string (e.g. "LEFT" or "RIGHT")
            nn_d:           the dynamics network (NNd)
            action_space:   ordered list of all legal actions (e.g. ["LEFT", "RIGHT"])

        Returns:
            next_sigma:     jnp array of shape [1, abstract_dim]
            reward_pred:    scalar float — predicted immediate reward
        """
        abstract_dim = abstract_state.shape[1]
        num_actions  = len(action_space)

        action_idx   = action_space.index(action)
        action_onehot = jnp.zeros(num_actions).at[action_idx].set(1.0)

        # Concatenate [σ, onehot(a)] → NNd input of shape [1, abstract_dim + num_actions]
        nnd_input = jnp.concatenate([abstract_state[0], action_onehot])[None, :]
        nnd_output = nn_d(nnd_input)[0]   # shape: [abstract_dim + 1]

        next_sigma  = nnd_output[:abstract_dim][None, :]   # [1, abstract_dim]
        reward_pred = float(nnd_output[-1])
        return next_sigma, reward_pred
