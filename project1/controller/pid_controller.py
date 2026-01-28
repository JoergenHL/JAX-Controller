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

    def step(self, params, error, sum_error, old_error):
        # Compute PID output: proportional + derivative + integral terms
        kp = params["kp"]
        kd = params["kd"]
        ki = params["ki"]
        dE = error - old_error

        U = kp*error + kd*dE + ki*sum_error
        
        return U
