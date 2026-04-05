from flax import nnx
import jax
import jax.numpy as jnp


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        self.w = nnx.Param(rngs.param.uniform((din, dout)))
        self.b = nnx.Param(jnp.zeros((dout, )))
        self.din, self.dout = din, dout

    def __call__(self, x: jax.Array):
        return x @ self.w + self.b  
    
class MLP(nnx.Module):
    def __init__(self, din: int, dout: int, hidden_dims: tuple, *, rngs: nnx.Rngs):
        """
        Args:
            din: Input dimension
            dout: Output dimension
            hidden_dims: Tuple of hidden layer dimensions, e.g., (16, 16, 16)
            rngs: Random number generators for initialization
        """
        # Store architecture info
        self.din = din
        self.dout = dout
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, tuple) else (hidden_dims,)
        
        # Build all layers using nnx.Dict for NNX compatibility
        self.layers = nnx.Dict()
        dims = [self.din] + list(self.hidden_dims) + [self.dout]
        
        for i in range(len(dims) - 1):
            self.layers[f"layer_{i}"] = Linear(dims[i], dims[i+1], rngs=rngs)

    def __call__(self, x: jax.Array):
        # Pass through all layers, applying activation after each except the last
        num_layers = len(self.layers)
        for i in range(num_layers):
            layer = self.layers[f"layer_{i}"]
            x = layer(x)
            if i < num_layers - 1:  # Don't apply activation to output layer
                x = nnx.gelu(x)
        return x
