
# "bathtube"
PLANT_TYPE = "bathtub"

# "pid"
CONTROLLER_TYPE = "pid"

PLANT_CONFIG = {
    "bathtub": {
        "H0": 1.,
        "A": 10.,
        "C": 10 / 100,
        "g": 9.8,
        "D": [-0.1, -0.05, 0, 0.05, 0.1],
        "T": 1.
    }
}

CONTROLLER_CONFIG = {
    "pid" : {
        "kp": 0.1,
        "kd": 0.1,
        "ki": 0.1,
        "lr": 0.1
    }
}

CONSYS_CONFIG = {
    "epochs": 1,
    "timesteps": 10,
    "seed": 42
}