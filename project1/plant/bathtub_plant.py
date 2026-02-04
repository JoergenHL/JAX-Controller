import jax.numpy as jnp
from .iplant import IPlant

# Bathtub water level dynamics with controlled inflow and gravity-driven outflow
class Bathtub_Plant(IPlant):

    def __init__(self, config):
        self.H0 = config["H0"]      # Initial height
        self.A = config["A"]        # Cross-sectional area
        self.C = config["C"]        # Outflow coefficient
        self.G = config["G"]        # Gravitational acceleration

    def init_state(self):
        return self.H0
    
    def get_velocity(self, H):
        # Torricelli's law: outflow velocity from height
        return jnp.sqrt(2 * self.G * H)
        
    def get_flow_loss(self, velocity):
        return velocity * self.C
    
    def get_dB(self, U, D, Q):
        # Net volume change: inflow - outflow
        return U + D - Q
    
    def get_dH(self, dB):
        # Convert volume change to height change
        return dB / self.A

    def step(self, H, U, D):
        # Simulate one timestep: control input U, disturbance D
        V = self.get_velocity(H)
        Q = self.get_flow_loss(V)
        dB = self.get_dB(U, D, Q)
        dH = self.get_dH(dB)
        new_H = jnp.clip(H + dH, 0)
        return new_H
    
    def output(self, state):
        return state

