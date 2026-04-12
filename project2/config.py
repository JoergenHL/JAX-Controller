# Central configuration for the AlphaZero / MuZero project.
# All hyperparameters live here so they can be changed in one place.

game = {
    "max_position": 5,   # Win at +5, lose at -5, start at 0
}

mcts = {
    "num_simulations": 20,  # Simulations per move (reduced from 50; 20 is sufficient for LineWorld)
    "c": 2,                 # Exploration constant in PUCT formula
    "d_max": 1,             # Max search depth in abstract space (no terminal check in u-MCTS)
    "dir_alpha":   0.3,     # Dirichlet concentration for root exploration noise
    "dir_epsilon": 0.25,    # Weight of noise vs network policy at root
}

nn = {
    # Stage 3: NNr encodes real states; NNp predicts from abstract states.
    # Stage 4A adds NNd: given abstract state + action, predict next abstract state + reward.
    #
    # NNr: real_state(1) → hidden(8) → abstract_state(4)
    # NNp: abstract_state(4) → hidden(16) → [value, logit_L, logit_R]
    # NNd: [abstract_state(4) ++ action_onehot(2)] → hidden(16) → [next_abstract(4), reward(1)]
    "nnr_dims": [1, 8, 4],
    "nnp_dims": [4, 16, 3],
    "nnd_dims": [6, 16, 5],   # input: 4 (abstract) + 2 (action onehot); output: 4 (next σ) + 1 (reward)
    "abstract_dim": 4,        # must match nnr_dims[-1] and nnp_dims[0] and nnd_dims[0]-num_actions
    "num_actions": 2,
    "learning_rate": 0.01,
}

training = {
    "num_iterations":    10,   # Self-play / train cycles
    "episodes_per_iter": 5,  # reduced from 20; fewer episodes, more iterations is faster
    "epochs_per_iter":   20,
    "roll_ahead":         3,   # w: steps to unroll NNd during BPTT 
}
