import jax
import jax.numpy as jnp
import numpy as np
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

    def __init__(self, consys_config):
        self.epochs = consys_config["epochs"]
        self.timesteps = consys_config["timesteps"]
        self.seed = consys_config["seed"]
        return

    
    def generate_noise(self, D, key):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(
            subkey,
            shape=(self.timesteps,),
            minval=0,
            maxval=len(D)
        )
        D_vals = jnp.array(D)
        noise_arr = D_vals[idx]
        return noise_arr

    
    def run_system(self):
        key = jax.random.PRNGKey(self.seed)

        plant_type = config.PLANT_TYPE
        plant_config = config.PLANT_CONFIG[plant_type]
        Plant_Class = PLANT_REGISTRY[plant_type]
        plant = Plant_Class(plant_config)

        controller_type = config.CONTROLLER_TYPE
        controller_config = config.CONTROLLER_CONFIG[controller_type]
        Controller_Class = CONTROLLER_REGISTRY[controller_type]
        controller = Controller_Class(controller_config)

        T = plant_config["T"]
        D = plant_config["D"]
        

        for k in range(self.epochs):
            noise_arr = self.generate_noise(D, key)
            

            print(f"Epoch: {k}")
            state = plant.init_state()
            controller.reset()

            for t in range(self.timesteps):
                print(f"Timestep: {t}")
                print(f"T: {T}")
                Y = plant.output(state)
                print(f"Y: {Y}")
                E = T - Y
                print(f"E: {E}")

                U = controller.step(E)
                print(f"U: {U}")

                noise = noise_arr[t]
                print(f"D: {noise}")

                state = plant.step(state, U, noise)



consys_config = config.CONSYS_CONFIG
system = Consys(consys_config)

system.run_system()
