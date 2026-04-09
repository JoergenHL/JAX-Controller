from flax import nnx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

import config
from nn.nn import MLP


class NNManager:
    """
    Network Manager (NNM)
    
    Responsibility: Own the networks and handle their training via BPTT.
    - Create networks
    - Retrieve networks
    - Train networks (BPTT: backpropagation through time)
    """
    
    def __init__(self):
        self.models = {}
    
    def create_net(self, name: str, dims: list[int]):
        """Create a network with given dimensions.
        
        Args:
            name: Identifier for this network
            dims: Layer dimensions [input, hidden1, hidden2, ..., output]
                  Example: [1, 16, 16, 2] → 1→16→16→2 network
        
        Returns:
            The created MLP model
        """
        model = MLP(dims, rngs=nnx.Rngs(1))
        self.models[name] = model
        return model
    
    def get_net(self, name: str):
        """Retrieve a network by name."""
        if name not in self.models:
            raise ValueError(f"Network '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]
    
    def bptt_train(self, trunk_name: str, value_name: str, training_data: list, 
                   num_epochs: int = 20, learning_rate: float = 0.01):
        """
        Train networks using BPTT (backpropagation through time).
        
        This is NNManager's core responsibility: update parameters via gradient descent.
        
        Args:
            trunk_name: Name of trunk network (uses fallback if not found)
            value_name: Name of value head network (uses fallback if not found)
            training_data: List of (state, target_return) tuples from episodes
            num_epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
        """
        if len(training_data) == 0:
            print("  WARNING: No training data provided!")
            return
        
        # Get networks with fallback for flexible naming
        try:
            trunk = self.get_net(trunk_name)
            value_head = self.get_net(value_name)
        except ValueError:
            # Fallback: use first two available networks
            available = list(self.models.keys())
            if len(available) < 2:
                raise ValueError(f"Need at least 2 networks, found: {available}")
            trunk = self.get_net(available[0])
            value_head = self.get_net(available[1])
        
        print(f"  Training on {len(training_data)} samples for {num_epochs} epochs...")
        
        # Prepare data
        states = jnp.array([[s] for s, _ in training_data], dtype=jnp.float32).squeeze()
        targets = jnp.array([t for _, t in training_data], dtype=jnp.float32)
        states_batch = states.reshape(-1, 1)
        
        # Training loop
        for epoch in range(num_epochs):
            # Compute loss
            trunk_out = trunk(states_batch)
            preds = value_head(trunk_out).squeeze()
            loss = jnp.mean((preds - targets) ** 2)
            
            # Update trunk layers (BPTT through multi-layer network)
            for i, layer in enumerate(trunk.layers):
                def loss_fn_trunk(w, b):
                    x = states_batch
                    for j, l in enumerate(trunk.layers):
                        if j == i:
                            x = jnp.dot(x, w) + b
                        else:
                            x = jnp.dot(x, l.w[...]) + l.b[...]
                        if j < len(trunk.layers) - 1:
                            x = jax.nn.relu(x)
                    pred = value_head(x).squeeze()
                    return jnp.mean((pred - targets) ** 2)
                
                loss_val, (w_grad, b_grad) = jax.value_and_grad(loss_fn_trunk, argnums=(0, 1))(
                    layer.w[...], layer.b[...]
                )
                layer.w[...] = layer.w[...] - 0.3 * learning_rate * w_grad
                layer.b[...] = layer.b[...] - 0.3 * learning_rate * b_grad
            
            # Update value head via BPTT
            layer = value_head.layers[-1]
            def loss_fn_value(w, b):
                trunk_features = trunk(states_batch)
                pred = jnp.dot(trunk_features, w) + b
                pred = pred.squeeze()
                return jnp.mean((pred - targets) ** 2)
            
            loss_val, (w_grad, b_grad) = jax.value_and_grad(loss_fn_value, argnums=(0, 1))(
                layer.w[...], layer.b[...]
            )
            layer.w[...] = layer.w[...] - learning_rate * w_grad
            layer.b[...] = layer.b[...] - learning_rate * b_grad
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{num_epochs}: loss = {float(loss_val):.6f}")



