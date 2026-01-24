import jax.numpy as jnp
from controller.icontroller import IController

# PID controller with proportional, derivative, and integral terms
class PID_Controller(IController):

    def __init__(self, params):
        self.error = 0
        self.sum_error = 0
        self.params = params
    
    def reset(self):
        # Reset error tracking
        self.error = 0
        self.sum_error = 0
        
    def update_params(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def step(self, error):
        # Compute PID output: proportional + derivative + integral terms
        kp = self.params["kp"]
        kd = self.params["kd"]
        ki = self.params["ki"]
        old_error = self.error
        dE = error - old_error

        U = kp*error + kd*dE + ki*self.sum_error

        self.sum_error += error
        self.error = error
        
        return U
