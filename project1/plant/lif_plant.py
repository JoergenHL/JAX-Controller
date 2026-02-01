from .iplant import IPlant
import jax.numpy as jnp

class LIF_Plant(IPlant):
    def __init__(self, config):
        self.spike_thr = config["spike_thr"]
        self.V0 = config["V0"]
        self.alfa = config["alfa"]
        self.gamma = config["gamma"]

    def init_state(self):
        return jnp.array([self.V0, jnp.array(0.0)])
    

    def step(self, state, U, D):
        V, r = state
        V_star = self.update_V(V, U, D)
        spike = jnp.where(V_star >= self.spike_thr, 1.0, 0.0)
        new_V = jnp.where(spike == 1.0, self.V0, V_star)
        new_r = self.update_r(r, spike)
        return jnp.array([new_V, new_r])

    def output(self, state):
        r = state[1]
        return r

    def update_V(self, V, U, D):
        return self.alfa * V + U + D
        
    def update_r(self, r, spike):
        return (1.0 - self.gamma) * r + self.gamma * spike
