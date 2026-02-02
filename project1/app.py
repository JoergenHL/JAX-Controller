import streamlit as st
import importlib
import copy
import matplotlib.pyplot as plt

import config
from consys import Consys


# ==================================================
# FREEZE BASELINE CONFIG (CRITICAL FIX)
# ==================================================
BASE_PLANT_TYPE = config.PLANT_TYPE
BASE_CONTROLLER_TYPE = config.CONTROLLER_TYPE

BASE_PLANT_CONFIG = copy.deepcopy(config.PLANT_CONFIG)
BASE_CONTROLLER_CONFIG = copy.deepcopy(config.CONTROLLER_CONFIG)
BASE_CONSYS_CONFIG = copy.deepcopy(config.CONSYS_CONFIG)


# ==================================================
# Utility: write full config.py
# ==================================================
def write_config(
    plant_type,
    controller_type,
    plant_config,
    controller_config,
    consys_config,
):
    content = f'''
# AUTO-GENERATED (temporary) CONFIG
# Written by Streamlit GUI

PLANT_TYPE = "{plant_type}"
CONTROLLER_TYPE = "{controller_type}"

PLANT_CONFIG = {plant_config}

CONTROLLER_CONFIG = {controller_config}

CONSYS_CONFIG = {consys_config}
'''
    with open("config.py", "w") as f:
        f.write(content)


# ==================================================
# Streamlit state guard
# ==================================================
if "run_requested" not in st.session_state:
    st.session_state.run_requested = False


# ==================================================
# SIDEBAR — GUI BUILT FROM BASE CONFIG ONLY
# ==================================================
st.sidebar.title("Experiment Configuration")

plant_type = st.sidebar.selectbox(
    "Plant Type",
    ["bathtub", "cournot", "lif"],
    index=["bathtub", "cournot", "lif"].index(BASE_PLANT_TYPE),
)

controller_type = st.sidebar.selectbox(
    "Controller Type",
    ["pid", "nn"],
    index=["pid", "nn"].index(BASE_CONTROLLER_TYPE),
)

plant_cfg = copy.deepcopy(BASE_PLANT_CONFIG[plant_type])
controller_cfg = copy.deepcopy(BASE_CONTROLLER_CONFIG[controller_type])
consys_cfg = copy.deepcopy(BASE_CONSYS_CONFIG)


# ==================================================
# Plant parameters
# ==================================================
st.sidebar.subheader("Plant Parameters")

if plant_type == "bathtub":
    plant_cfg["H0"] = st.sidebar.number_input("H0", value=float(plant_cfg["H0"]))
    plant_cfg["A"] = st.sidebar.number_input("A", value=float(plant_cfg["A"]))
    plant_cfg["C"] = st.sidebar.number_input("C", value=float(plant_cfg["C"]))
    plant_cfg["G"] = st.sidebar.number_input("G", value=float(plant_cfg["G"]))
    plant_cfg["T"] = st.sidebar.number_input("Target T", value=float(plant_cfg["T"]))

elif plant_type == "cournot":
    plant_cfg["p_max"] = st.sidebar.number_input("p_max", value=float(plant_cfg["p_max"]))
    plant_cfg["cm"] = st.sidebar.number_input("cm", value=float(plant_cfg["cm"]))
    plant_cfg["init_vals"] = st.sidebar.number_input(
        "Initial q", value=float(plant_cfg["init_vals"])
    )
    plant_cfg["T"] = st.sidebar.number_input("Target T", value=float(plant_cfg["T"]))

elif plant_type == "lif":
    plant_cfg["spike_thr"] = st.sidebar.number_input(
        "Spike threshold", value=float(plant_cfg["spike_thr"])
    )
    plant_cfg["V0"] = st.sidebar.number_input(
        "Reset voltage V0", value=float(plant_cfg["V0"])
    )
    plant_cfg["alfa"] = st.sidebar.slider(
        "Alpha (leak)", 0.7, 0.99, float(plant_cfg["alfa"])
    )
    plant_cfg["gamma"] = st.sidebar.slider(
        "Gamma (rate smoothing)", 0.01, 0.9, float(plant_cfg["gamma"])
    )
    plant_cfg["T"] = st.sidebar.slider(
        "Target firing rate", 0.0, 1.0, float(plant_cfg["T"])
    )


# ==================================================
# Controller parameters
# ==================================================
st.sidebar.subheader("Controller Parameters")

if controller_type == "pid":
    controller_cfg["kp"] = st.sidebar.number_input("kp", value=float(controller_cfg["kp"]))
    controller_cfg["ki"] = st.sidebar.number_input("ki", value=float(controller_cfg["ki"]))
    controller_cfg["kd"] = st.sidebar.number_input("kd", value=float(controller_cfg["kd"]))

elif controller_type == "nn":
    layers_str = st.sidebar.text_input(
        "NN layers (comma-separated)",
        value=",".join(map(str, controller_cfg["layers"])),
    )
    controller_cfg["layers"] = [int(x) for x in layers_str.split(",")]

    controller_cfg["activation_func"] = st.sidebar.selectbox(
        "Activation",
        ["relu", "tanh", "sigmoid"],
        index=["relu", "tanh", "sigmoid"].index(controller_cfg["activation_func"]),
    )

    wmin, wmax = controller_cfg["param_range"]
    controller_cfg["param_range"] = st.sidebar.slider(
        "Weight init range", -0.1, 0.1, (float(wmin), float(wmax))
    )


# ==================================================
# Training parameters
# ==================================================
st.sidebar.subheader("Training Parameters")

consys_cfg["epochs"] = st.sidebar.number_input(
    "Epochs", value=int(consys_cfg["epochs"]), step=100
)
consys_cfg["timesteps"] = st.sidebar.number_input(
    "Timesteps", value=int(consys_cfg["timesteps"])
)
consys_cfg["seed"] = st.sidebar.number_input(
    "Random seed", value=int(consys_cfg["seed"])
)
consys_cfg["lr"] = st.sidebar.number_input(
    "Learning rate", value=float(consys_cfg["lr"])
)

D_str = st.sidebar.text_input(
    "Noise values (comma-separated)",
    value=",".join(map(str, consys_cfg["D"]))
)
consys_cfg["D"] = [float(x) for x in D_str.split(",")]


# ==================================================
# MAIN PANEL
# ==================================================
st.title("JAX Controller")
st.markdown("Configure parameters in the sidebar, then run an experiment.")

if st.button("Run experiment"):
    st.session_state.run_requested = True


# ==================================================
# EXECUTION (ISOLATED)
# ==================================================
if st.session_state.run_requested:
    st.session_state.run_requested = False

    plt.close("all")

    # --- Build FULL config (preserve all entries) ---
    full_plant_config = copy.deepcopy(BASE_PLANT_CONFIG)
    full_controller_config = copy.deepcopy(BASE_CONTROLLER_CONFIG)

    full_plant_config[plant_type] = plant_cfg
    full_controller_config[controller_type] = controller_cfg

    # --- Write temporary execution config ---
    write_config(
        plant_type,
        controller_type,
        full_plant_config,
        full_controller_config,
        consys_cfg,
    )

    importlib.reload(config)

    # --- Run system ---
    system = Consys(config.CONSYS_CONFIG)
    fig = system.run_system()
    st.pyplot(fig)

    # --- Restore BASE config (CRITICAL) ---
    write_config(
        BASE_PLANT_TYPE,
        BASE_CONTROLLER_TYPE,
        BASE_PLANT_CONFIG,
        BASE_CONTROLLER_CONFIG,
        BASE_CONSYS_CONFIG,
    )

    st.success("Run completed.")
