# Central configuration for the AlphaZero / MuZero project.
# All hyperparameters live here so they can be changed in one place.

game = {
    "max_position": 5,   # Win at +5, lose at -5, start at 0
}

mcts = {
    "num_simulations": 50,  # Simulations per move (must reach depth 5 for LineWorld)
    "c": 2,                 # Exploration constant in PUCT formula
}

nn = {
    # Stage 3: two separate networks — NNr encodes real states, NNp predicts from abstract states.
    # NNr: real_state(1) → hidden(8) → abstract_state(4)
    # NNp: abstract_state(4) → hidden(16) → [value, logit_L, logit_R]
    "nnr_dims": [1, 8, 4],
    "nnp_dims": [4, 16, 3],
    "learning_rate": 0.01,
}

training = {
    "num_iterations": 10,   # Self-play / train cycles
    "episodes_per_iter": 20,
    "epochs_per_iter": 20,
}
