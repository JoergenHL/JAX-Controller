# JAX Controller - Control System Learning

A flexible control system framework using JAX for gradient-based learning of PID and neural network controllers across different plant models.

## Overview

This project implements a control system training pipeline where:
- **Plants**: Different dynamical systems to control (Bathtub water level, Cournot competition model, LIF neuron)
- **Controllers**: Parameterized control strategies (PID, Neural Networks) that learn to minimize error
- **Learning**: JAX-based automatic differentiation to optimize controller parameters through gradient descent

## Project Structure

```
project1/
├── plant/                 # Plant models (systems to control)
│   ├── iplant.py         # Plant interface
│   ├── bathtub_plant.py  # Water tank filling/draining
│   ├── cournot_plant.py  # Economic competition model
│   └── lif_plant.py      # Leaky Integrate-and-Fire neuron
├── controller/           # Control strategies
│   ├── icontroller.py    # Controller interface
│   ├── pid_controller.py # PID controller
│   └── nn_controller.py  # Neural network controller
├── consys.py            # Main training system
├── config.py            # Configuration (plants, controllers, training)
├── app.py               # Streamlit interactive GUI
```

## Installation

### Requirements
- Python 3.8+
- JAX
- Matplotlib
- Streamlit (for GUI)

### Setup

```bash
# Create environment
conda create -n control-system python=3.9

# Activate environment
conda activate control-system

# Install dependencies
pip install jax jaxlib matplotlib streamlit
```

## Usage

### Command Line (Direct Training)

Run a training experiment with current config:

```bash
python consys.py
```

This will:
1. Train the controller for 1000 epochs
2. Display learning progression plot showing loss and parameter evolution
3. Save the figure

### Interactive GUI

Launch the Streamlit app for interactive experiments:

```bash
streamlit run app.py
```

Features:
- Select plant type (bathtub, cournot, lif)
- Select controller type (pid, nn)
- Adjust all plant, controller, and training parameters in real-time
- Run experiments and visualize results
- Configuration changes are isolated per run

## Configuration

Edit `config.py` to set experiment parameters:

### Plants

```python
PLANT_CONFIG = {
    "bathtub": {
        "H0": 1.,              # Initial water height
        "A": 10.,             # Tank cross-section area
        "C": 10 / 100,        # Drain coefficient
        "G": 9.8,             # Gravity
        "T": 1.               # Target height
    },
    "cournot": {
        "p_max": 5.,          # Max price
        "cm": 0.1,            # Marginal cost
        "init_vals": 0.1,     # Initial quantity
        "T": 3.               # Target price
    },
    "lif": {
        "spike_thr": -0.05,   # Voltage threshold for spikes
        "V0": -0.07,          # Reset voltage after spike
        "alfa": 0.7,          # Leak coefficient (decay)
        "gamma": 0.2,         # Firing rate smoothing
        "T": 0.5              # Target firing rate
    }
}
```

### Controllers

```python
CONTROLLER_CONFIG = {
    "pid": {
        "kp": 0.1,             # Proportional gain
        "ki": 0.1,             # Integral gain
        "kd": 0.1              # Derivative gain
    },
    "nn": {
        "layers": [3, 16, 16, 16, 1],     # Network architecture
        "activation_func": "relu",         # relu, tanh, or sigmoid
        "param_range": (0.0, 0.1)         # Weight initialization range
    }
}
```

### Training

```python
CONSYS_CONFIG = {
    "epochs": 1000,        # Training iterations
    "timesteps": 100,      # Steps per epoch
    "seed": 42,            # Random seed
    "lr": 0.001,           # Learning rate
    "D": [-0.01, -0.005, 0, 0.005, 0.01]  # Noise/disturbance values
}
```

## How It Works

### Training Loop

1. **Forward Pass**: Run the plant for `timesteps` iterations
   - Track error at each step
   - Controller produces control input based on PID or NN
   - Plant state updates with control input and disturbance

2. **Loss Calculation**: Mean squared error over all timesteps

3. **Gradient Computation**: JAX computes gradients of loss w.r.t. controller parameters

4. **Parameter Update**: Gradient descent: `param = param - lr * gradient`

### Key Implementation Details

- **JAX Scan**: Efficiently rolls out timesteps using `jax.lax.scan`
- **JIT Compilation**: Performance boost with `jax.jit`
- **Differentiable Spike**: LIF neuron uses smooth sigmoid spike instead of hard threshold to allow gradient flow

## Example Results

### PID Learning on LIF Plant
- **kp** (proportional gain) increases over time
- **kd** (derivative gain) increases to improve response
- **ki** (integral gain) remains small due to error elimination
- Loss decreases with training

### NN Learning on Bathtub Plant
- Network learns implicit control strategy
- Handles disturbances through learned patterns
- Typically achieves lower loss than PID on complex plants


## Extending the Project

### Add a New Plant

1. Create `plant/my_plant.py`
2. Inherit from `IPlant` interface
3. Implement: `init_state()`, `step()`, `output()`
4. Register in `config.py` and `consys.py`

### Add a New Controller

1. Create `controller/my_controller.py`
2. Inherit from `IController` interface
3. Implement: `get_params()`, `step()`
4. Register in `config.py` and `consys.py`


## License

Educational project

## References

- JAX documentation: https://jax.readthedocs.io/
