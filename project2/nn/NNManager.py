from flax import nnx
import jax.numpy as jnp
import jax
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from nn.nn import MLP


class NNManager:
    def __init__(self):
        self.model = None
    
    def create_model(self, din: int, dout: int):
        """Create a model from config specifications."""
        hidden_dims = config.nn["hidden"]
        self.model = MLP(din, dout, hidden_dims, rngs=nnx.Rngs(1))
        return self.model
