from .icontroller import IController
import jax.numpy as jnp
import jax
import jax.nn as jnn

# Neural network based controller
class NN_Controller(IController):

    def __init__(self, config, key):
        # Initialize neural network configuration
        self.layers = config["layers"]
        self.activation_func = self.resolve_activation_func(config["activation_func"])
        self.param_range = config["param_range"]
        # Initialize network weights and biases
        self.params = self.init_nn(
            layers=self.layers,
            key=key
        )

    def resolve_activation_func(self, name):
        # Select activation function by name
        if name == "sigmoid":
            return jnn.sigmoid
        elif name == "tanh":
            return jnn.tanh
        elif name == "relu":
            return jnn.relu
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def init_nn(self, layers, key):
        # Initialize network parameters (weights and biases)
        params = []

        # Split random key for each layer
        keys = jax.random.split(key, len(layers) - 1)

        # Initialize each layer
        for k , key in enumerate(keys):
            in_dim = layers[k]
            out_dim = layers[k+1]

            # Split key for weights and biases
            key_W, key_b = jax.random.split(key)

            # Initialize weights and biases with random uniform distribution
            W = jax.random.uniform(key_W, shape=(out_dim, in_dim), minval=self.param_range[0], maxval=self.param_range[1])
            b = jax.random.uniform(key_b, shape=(out_dim,), minval=self.param_range[0], maxval=self.param_range[1] )

            params.append((W, b))

        return params

    def get_params(self):
        # Return network parameters
        return self.params
    
    def step(self, params, errors):
        # Extract error signals
        error, sum_error, old_error = errors
        # Calculate error derivative
        dE = error - old_error  

        # Create network input from errors (proportional, derivative, integral)
        x = jnp.array([error, dE, sum_error])
        
        # Forward pass through hidden layers
        for (W, b) in params[:-1]:
            x = self.activation_func(W @ x + b)

        # Output layer (no activation)
        W_out, b_out = params[-1]
        U = W_out @ x + b_out
        return jnp.squeeze(U)