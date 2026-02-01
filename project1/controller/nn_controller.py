from .icontroller import IController
import jax.numpy as jnp
import jax
import jax.nn as jnn

class NN_Controller(IController):

    def __init__(self, config, key):
        self.layers = config["layers"]
        self.activation_func = self.resolve_activation_func(config["activation_func"])
        self.param_range = config["param_range"]
        self.params = self.init_nn(
            layers=self.layers,
            key=key
        )

    def resolve_activation_func(self, name):
        if name == "sigmoid":
            return jnn.sigmoid
        elif name == "tanh":
            return jnn.tanh
        elif name == "relu":
            return jnn.relu
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def init_nn(self, layers, key):
        params = []

        keys = jax.random.split(key, len(layers) - 1)

        for k , key in enumerate(keys):
            in_dim = layers[k]
            out_dim = layers[k+1]

            key_W, key_b = jax.random.split(key)

            W = jax.random.uniform(key_W, shape=(out_dim, in_dim), minval=self.param_range[0], maxval=self.param_range[1])
            b = jax.random.uniform(key_b, shape=(out_dim,), minval=self.param_range[0], maxval=self.param_range[1] )

            params.append((W, b))

        return params

    def get_params(self):
        return self.params
    
    def step(self, params, errors):
        error, sum_error, old_error = errors
        dE = error - old_error  

        x = jnp.array([error, dE, sum_error])
        
        for (W, b) in params[:-1]:
            x = self.activation_func(W @ x + b)

        W_out, b_out = params[-1]
        U = W_out @ x + b_out
        return jnp.squeeze(U)