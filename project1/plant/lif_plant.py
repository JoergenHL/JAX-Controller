from .iplant import IPlant
import jax.numpy as jnp
import jax.nn as jnn

# Leaky Integrate-and-Fire neuron model
class LIF_Plant(IPlant):
    def __init__(self, config):
        # Initialize LIF neuron parameters
        self.spike_thr = config["spike_thr"]
        self.V0 = config["V0"]
        self.alfa = config["alfa"]
        self.gamma = config["gamma"]

    def init_state(self):
        # Initialize voltage and firing rate
        return jnp.array([self.V0, 0.0])
    
    def step(self, state, U, D):
        # Update neuron state: voltage, spikes, and firing rate
        V, r = state
        external_input = self.external_input(D)
        V_star = self.update_V(V, U, external_input)
        spike = jnn.sigmoid((V_star - self.spike_thr) * 10) 
        new_V = jnp.where(spike == 1.0, self.V0, V_star)
        new_r = self.update_r(r, spike)
        return jnp.array([new_V, new_r])

    def output(self, state):
        # Return firing rate as system output
        r = state[1]
        return r

    def update_V(self, V, U, external_input):
        # Leaky integration: new voltage with decay and input
        return self.alfa * V + U + external_input
        
    def update_r(self, r, spike):
        # Update firing rate based on spike activity
        return (1.0 - self.gamma) * r + self.gamma * spike
    
    def external_input(self, D):
        # Scale disturbance to input signal
        return D 
