import jax
import jax.numpy as jnp
import config

from plant.bathtub_plant import Bathtub_Plant
from controller.pid_controller import PID_Controller

PLANT_REGISTRY = {
    "bathtub": Bathtub_Plant
}

CONTROLLER_REGISTRY = {
    "pid": PID_Controller
}

class Consys():

    def __init__():
        return
    
    def run_system():
        plant_type = config.PLANT_TYPE
        plant_config = config.PLANT_CONFIG
        Plant_Class = PLANT_REGISTRY[plant_type]
        plant = Plant_Class(plant_config)

        controller_type = config.CONTROLLER_TYPE
        controller_config = config.CONTROLLER_CONFIG[controller_type]
        Controller_Class = CONTROLLER_REGISTRY[controller_type]
        controller = Controller_Class(controller_config)

        for m in epoch:


            state = plant.init_state()

            for t in range(num_timesteps):
                Y = plant.output
                E = Target - Y
                U = controller(E)
                state = plant.step(state, U, D[t])

            