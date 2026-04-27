from flax import nnx
import jax
import jax.numpy as jnp
import math


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        # He initialization: std = sqrt(2 / fan_in), designed for ReLU networks.
        # Ensures activations have unit variance after each layer so gradients
        # don't vanish as they propagate backwards through NNr → NNd → NNp.
        # With 0.01 * normal, all abstract states collapse near zero and MCTS
        # gets no signal to distinguish positions.
        self.w = nnx.Param(rngs.param.normal((din, dout)) * math.sqrt(2.0 / din))
        self.b = nnx.Param(jnp.zeros((dout, )))

    def __call__(self, x: jax.Array):
        return x @ self.w + self.b


class MLP(nnx.Module):
    """Simple MLP: takes layer dimensions and builds network.
    
    Example: MLP([8, 16, 16, 4], rngs) creates 8→16→16→4 net
    with ReLU between layers, no activation on output.
    """
    
    def __init__(self, dims: list[int], *, rngs: nnx.Rngs):
        """
        Args:
            dims: List of dimensions [input, hidden1, hidden2, ..., output]
            rngs: Random number generators
        """
        self.layers = nnx.List(
            [Linear(dims[i], dims[i+1], rngs=rngs) for i in range(len(dims) - 1)]
        )

    def __call__(self, x: jax.Array):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on last layer
                x = nnx.relu(x)
        return x




