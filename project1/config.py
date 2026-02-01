
# "bathtube", "cournot"
PLANT_TYPE = "cournot"

# "pid", "nn"
CONTROLLER_TYPE = "pid"

PLANT_CONFIG = {
    "bathtub": {
        "H0": 1.,
        "A": 10.,
        "C": 10 / 100,
        "G": 9.8,
        "T": 1.
    },

    "cournot": {
        "p_max": 5.,
        "cm": 0.1,
        "T": 3.,
        "init_vals": 0.1
    }

}

CONTROLLER_CONFIG = {
    "pid" : {
        "kp": 0.1,
        "kd": 0.1,
        "ki": 0.1
    },

    "nn" : {
        "layers": [3, 16, 16, 16, 1],
        "activation_func": "relu", # sigmoid, tanh, relu
        "param_range": (-0.01, 0.01),
    }
}

CONSYS_CONFIG = {
    "epochs": 100,
    "timesteps": 30,
    "seed": 42,
    "lr": 0.01,
    "D": [-0.01, -0.005, 0, 0.005, 0.01]
}