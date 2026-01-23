
# "bathtube"
PLANT_TYPE = "bathtube"

# "pid"
CONTROLLER_TYPE = "pid"

PLANT_CONFIG = {
    "bathtub": {
        "H0": 1.,
        "A": 10.,
        "C": 10 / 100,
        "g": 9.8,
        "D": [-0.1, -0.05, 0, 0.05, 0.1]
    }
}

CONTROLLER_CONFIG = {
    "pid" : {
        "kp": 0.,
        "kd": 0.,
        "ki": 0.
    }
}