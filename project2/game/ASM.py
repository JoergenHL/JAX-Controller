import jax.numpy as jnp


class ASM:
    """Abstract State Manager — bridges real game states and abstract (latent) states.

    In MuZero, all planning happens in abstract space. The ASM is the only place
    that knows how to move between the two worlds:

        real state  ──NNr──>  abstract state σ  ──NNp──>  (value, policy)
                                                  ──NNd──>  (next σ, reward)  [Stage 4]

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

    def next_abstract_state(self, abstract_state, action, nn_d):
        """Predict next abstract state and reward via NNd.

        Stage 4 — NNd not yet implemented.
        When implemented: (σ, action) → (next_σ, predicted_reward)
        """
        return None
