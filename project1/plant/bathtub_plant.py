
import jax.numpy as jnp
from iplant import IPlant

# Bathtub model with water level dynamics
class Bathtub_Plant(IPlant):

    def __init__(self, config):
        self.H0 = config["H0"]
        self.A = config["A"]
        self.C = config["C"]
        self.g = config["g"]

    def init_state(self):
        return self.H0
    
    # Outflow velocity from height 
    def get_velocity(self, H): 
        return jnp.sqrt(2 * self.g * H)
        
    # Outflow rate
    def get_flow_loss(self, velocity): 
        return velocity * self.C
    
    # Change in volume
    def get_dB(self, U, D, Q):
        return  U + D - Q
    
    # Change in height
    def get_dH(self, dB):
        return dB / self.A

    def step(self, H, U, D):
        V = self.get_velocity(H)
        Q = self.get_flow_loss(V)
        dB = self.get_dB(U, D, Q)
        dH = self.get_dH(dB)
        new_H = H + dH
        return new_H
    
    def output(self, state):
        return state

