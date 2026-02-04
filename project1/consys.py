import jax
import jax.numpy as jnp
import config
import matplotlib.pyplot as plt

# Import plant classes
from plant.bathtub_plant import Bathtub_Plant
from plant.cournot_plant import Cournot_Plant
from plant.lif_plant import LIF_Plant

# Import controller classes
from controller.pid_controller import PID_Controller
from controller.nn_controller import NN_Controller

# Registry for available plants
PLANT_REGISTRY = {
    "bathtub": Bathtub_Plant,
    "cournot": Cournot_Plant,
    "lif": LIF_Plant
}

# Registry for available controllers
CONTROLLER_REGISTRY = {
    "pid": PID_Controller,
    "nn": NN_Controller
}

# Run one full epoch of training
def run_one_epoch(params, controller, noise_arr, plant, target):
        # Initialize plant state
        state = plant.init_state()

        # Initialize error terms for control loop
        iE = jnp.array(0.0)  # integral error
        old_E = jnp.array(0.0)  # previous error

        init_carry = (state, iE, old_E)
        input = noise_arr

        # Execute one timestep of the system
        def run_one_timestep(carry, noise_t):
            state, iE, old_E = carry

            # Get plant output and calculate error
            Y = plant.output(state)
            E = target - Y
            iE = iE + E  # accumulate integral error
            
            # Calculate control input
            U = controller.step(params, (E, iE, old_E))
            old_E = E
            
            # Update plant state
            new_state = plant.step(state, U, noise_t)
            
            return (new_state, iE, old_E), E**2
        
        # Run all timesteps using JAX scan
        (final_carry, error_history) = jax.lax.scan(
            run_one_timestep,
            init_carry, 
            input
        )

        # Calculate MSE loss
        loss = jnp.sqrt(jnp.mean(error_history))

        return loss

# Main control system class
class Consys():

    def __init__(self, consys_config):
        # Initialize control system parameters
        self.epochs = consys_config["epochs"]
        self.timesteps = consys_config["timesteps"]
        self.seed = consys_config["seed"]
        self.lr = consys_config["lr"]
        self.D = consys_config["D"]
        return

    
    def generate_noise(self, D, key):
        # Generate random noise array for one epoch
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
        # Initialize random key
        key = jax.random.PRNGKey(self.seed)
        
        # Load plant from config
        plant_type = config.PLANT_TYPE
        plant_config = config.PLANT_CONFIG[plant_type]
        Plant_Class = PLANT_REGISTRY[plant_type]
        plant = Plant_Class(plant_config)

        # Load controller from config
        controller_type = config.CONTROLLER_TYPE
        controller_config = config.CONTROLLER_CONFIG[controller_type]
        Controller_Class = CONTROLLER_REGISTRY[controller_type]
        controller = Controller_Class(controller_config, key)

        # Get initial controller parameters
        params = controller.get_params()

        # Compile functions with JAX for performance
        run_one_epoch_jit = jax.jit(run_one_epoch, static_argnums=(1, 3))
        gradfunc_jit = jax.grad(run_one_epoch_jit, argnums=0)
        
        # Get target and disturbance values
        T = plant_config["T"]
        D = self.D

        # Track losses and PID parameters if using PID controller
        losses = []
        pid = False
        if controller_type == "pid":
            pid = True
            kp = []
            kd = []
            ki = []
        
        # Training loop
        for k in range(self.epochs):
            # Generate noise for this epoch
            noise_arr, key = self.generate_noise(D, key)

            # Run epoch and calculate gradients
            loss = run_one_epoch_jit(params, controller, noise_arr=noise_arr, plant=plant, target=T)
            grads = gradfunc_jit(params, controller, noise_arr=noise_arr, plant=plant, target=T)

            # Update parameters using gradient descent
            params = jax.tree.map(
                 lambda p, g: p - self.lr * g, 
                 params, 
                 grads
            )
            
            # Store parameters if using PID
            if pid:
                kp.append(params["kp"])
                kd.append(params["kd"])
                ki.append(params["ki"])
            losses.append(loss)

        # Plot results
        plt.figure(figsize=(10,5))
        plt.suptitle(f"Learning for {controller_type.upper()}-controller")
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Learning Progression")
        
        # Plot PID parameters if applicable
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
        fig = plt.gcf()
        plt.close(fig)
        return fig

# Initialize and run the control system
consys_config = config.CONSYS_CONFIG
system = Consys(consys_config)

system.run_system()
