
# "bathtube", "cournot", "lif"
PLANT_TYPE = "bathtub"

# "pid", "nn"
CONTROLLER_TYPE = "nn"

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
    },

    "lif": {
        "spike_thr": -0.05,
        "V0": -0.07,
        "alfa": 0.7,
        "gamma": 0.2,
        "T": 0.5

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
        "param_range": (0.0, 0.1),
    }
}

CONSYS_CONFIG = {
    "epochs": 1000,
    "timesteps": 100,
    "seed": 42,
    "lr": 0.001,
    "D": [-0.01, -0.005, 0, 0.005, 0.01]
}