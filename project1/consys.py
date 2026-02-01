import jax
import jax.numpy as jnp
import config
import matplotlib.pyplot as plt

from plant.bathtub_plant import Bathtub_Plant
from plant.cournot_plant import Cournot_Plant
from plant.lif_plant import LIF_Plant
from controller.pid_controller import PID_Controller
from controller.nn_controller import NN_Controller

PLANT_REGISTRY = {
    "bathtub": Bathtub_Plant,
    "cournot": Cournot_Plant,
    "lif": LIF_Plant
}

CONTROLLER_REGISTRY = {
    "pid": PID_Controller,
    "nn": NN_Controller
}

def run_one_epoch(params, controller, noise_arr, plant, target):
        state = plant.init_state()

        iE = jnp.array(0.0)
        old_E = jnp.array(0.0)

        init_carry = (state, iE, old_E)
        input = noise_arr

        def run_one_timestep(carry, noise_t):
            state, iE, old_E = carry

            Y = plant.output(state)
            E = target - Y
            iE = iE + E
            U = controller.step(params, (E, iE, old_E))
            old_E = E
            new_state = plant.step(state, U, noise_t)
            
            return (new_state, iE, old_E), E**2
        
        (final_carry, error_history) = jax.lax.scan(
            run_one_timestep,
            init_carry, 
            input
        )

        loss = jnp.sqrt(jnp.mean(error_history))

        return loss

class Consys():

    def __init__(self, consys_config):
        self.epochs = consys_config["epochs"]
        self.timesteps = consys_config["timesteps"]
        self.seed = consys_config["seed"]
        self.lr = consys_config["lr"]
        self.D = consys_config["D"]
        return

    
    def generate_noise(self, D, key):
        # This is where timesteps are used, later noise_arr define the timesteps in lax.scan
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(
            subkey,
            shape=(self.timesteps,),
            minval=0,
            maxval=len(D)
        )
        D_vals = jnp.array(D)
        noise_arr = D_vals[idx]
        return noise_arr, key


    def run_system(self):
        key = jax.random.PRNGKey(self.seed)
        plant_type = config.PLANT_TYPE
        plant_config = config.PLANT_CONFIG[plant_type]
        Plant_Class = PLANT_REGISTRY[plant_type]
        plant = Plant_Class(plant_config)

        controller_type = config.CONTROLLER_TYPE
        controller_config = config.CONTROLLER_CONFIG[controller_type]
        Controller_Class = CONTROLLER_REGISTRY[controller_type]
        controller = Controller_Class(controller_config, key)

        params = controller.get_params()

        # gradfunc = jax.grad(run_one_epoch, argnums=0)

        run_one_epoch_jit = jax.jit(run_one_epoch, static_argnums=(1, 3))
        gradfunc_jit =jax.grad(run_one_epoch_jit, argnums=0)
        
        T = plant_config["T"]
        D = self.D

        losses = []
        pid = False
        if controller_type == "pid":
            pid = True
            kp = []
            kd = []
            ki = []
        
        for k in range(self.epochs):
            noise_arr, key = self.generate_noise(D, key)

            loss = run_one_epoch_jit(params, controller, noise_arr=noise_arr, plant=plant, target=T)
            #loss = run_one_epoch(params, controller, noise_arr=noise_arr, plant=plant, target=T)
            grads = gradfunc_jit(params, controller, noise_arr=noise_arr, plant=plant, target=T)

            params = jax.tree.map(
                 lambda p, g: p - self.lr * g, 
                 params, 
                 grads
            )
            
            if pid:
                kp.append(params["kp"])
                kd.append(params["kd"])
                ki.append(params["ki"])
            losses.append(loss)

            if k % 100 == 0:
                print(f"Loss for epoch {k}: {loss}")

        plt.figure(figsize=(10,5))
        plt.suptitle(f"Learning for {controller_type.upper()}-controller")
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Learning Progression")
        
        if pid:
            plt.subplot(1, 2, 2)
            plt.plot(kp, label="kp")
            plt.plot(kd, label="kd")
            plt.plot(ki, label="ki")
            plt.xlabel("Epoch")
            plt.ylabel("Y")
            plt.legend()
            plt.title("Control Parameters")

        plt.show()
        
consys_config = config.CONSYS_CONFIG
system = Consys(consys_config)

system.run_system()
