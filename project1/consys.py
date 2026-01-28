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

def run_one_epoch(params, controller, noise_arr, plant, target, timesteps):
        controller.reset()
        state = plant.init_state()

        errors = jnp.zeros(timesteps)
        iE = jnp.array(0.0)
        old_E = jnp.array(0.0)

        for t in range(timesteps):
            Y = plant.output(state)
            E = target - Y
            iE += E
            U = controller.step(params, E, iE, old_E)
            old_E = E

            noise = noise_arr[t]
            state = plant.step(state, U, noise)

            errors = errors.at[t].set(E**2)

        loss = jnp.sqrt(jnp.mean(errors))

        return loss

class Consys():

    def __init__(self, consys_config):
        self.epochs = consys_config["epochs"]
        self.timesteps = consys_config["timesteps"]
        self.seed = consys_config["seed"]
        self.lr = consys_config["lr"]
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

        params = controller.get_params()

        T = plant_config["T"]
        D = plant_config["D"]
        timesteps = self.timesteps

        gradfunc = jax.grad(run_one_epoch, argnums=0)

        for k in range(self.epochs):
            noise_arr = self.generate_noise(D, key)

            loss = run_one_epoch(params, controller, noise_arr=noise_arr, plant=plant, target=T, timesteps=timesteps)
            """ state = plant.init_state()
            controller.reset()

            for t in range(self.timesteps):
                Y = plant.output(state)
                E = T - Y
                U = controller.step(E)
                noise = noise_arr[t]
                state = plant.step(state, U, noise)

                sum_E += E """
            
            grads = gradfunc(params, controller, noise_arr=noise_arr, plant=plant, target=T, timesteps=timesteps)

            params = jax.tree.map(
                 lambda p, g: p - self.lr * g, 
                 params, 
                 grads
            )

            if k % 10 == 0:
                print(f"Loss: {loss}")
                print(f"Params: {params}")





consys_config = config.CONSYS_CONFIG
system = Consys(consys_config)

system.run_system()

