from .iplant import IPlant
import jax.numpy as jnp

# Cournot duopoly: two firms competing on production quantity
class Cournot_Plant(IPlant):
    def __init__(self, config):
        self.p_max = config["p_max"]
        self.cm = config["cm"]
        self.T = config["T"]
        self.init_vals = config["init_vals"]
    
    def init_state(self):
        # Both firms start with initial quantity
        return jnp.array([self.init_vals, self.init_vals])

    def step(self, state, U, D):
        # Update quantities with control inputs, clip to [0, 1]
        q1, q2 = state
        new_q1 = jnp.clip(self.update_q1(U, q1), 0, 1)
        new_q2 = jnp.clip(self.update_q2(D, q2), 0, 1)
        return jnp.array([new_q1, new_q2])

    def output(self, state):
        # Return firm 1's profit
        q1, q2 = state
        q = self.production(q1, q2)
        price = self.price(q)
        profit = self.compute_profit(q1, price)
        return profit
    
    def production(self, q1, q2):
        return q1 + q2

    def price(self, q):
        # Inverse demand: price decreases with total quantity
        return self.p_max - q
    
    def update_q1(self, U, q1):
        return U + q1
    
    def update_q2(self, D, q2):
        return D + q2
    
    def compute_profit(self, q1, price):
        return q1 * (price - self.cm)
    

