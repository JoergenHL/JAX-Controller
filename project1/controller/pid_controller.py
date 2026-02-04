from .icontroller import IController

# PID controller with proportional, derivative, and integral terms
class PID_Controller(IController):

    def __init__(self, config, key=None):
        self.params = config

    def get_params(self):
        return self.params

    def step(self, params, errors):
        # Compute PID output: proportional + derivative + integral terms
        error, sum_error, old_error = errors
        kp = params["kp"]
        kd = params["kd"]
        ki = params["ki"]
        dE = error - old_error

        U = kp*error + kd*dE + ki*sum_error
        
        return U
