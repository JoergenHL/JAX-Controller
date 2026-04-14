# Central configuration for the AlphaZero / MuZero project.
# All hyperparameters live here so they can be changed in one place.
#
# Network dimensions are NOT stored here — they are derived at startup from
# the game object (game.state_dim, game.num_actions) and the tuning knobs below.
#
#   NNr full dims: [state_dim]              + nnr_hidden + [abstract_dim]
#   NNp full dims: [abstract_dim]           + nnp_hidden + [1 + num_actions]
#   NNd full dims: [abstract_dim+num_actions] + nnd_hidden + [abstract_dim + 1]
#
# To change network depth/width, edit the hidden lists below (e.g. [128, 64]).
# To swap games, change the game import in train_system.py — nothing else.

mcts = {
    "num_simulations": 10,   # Simulations per move
    "c": 2,                  # Exploration constant in PUCT formula
    "d_max": 1,              # Rollout depth: NNd steps from the randomly-picked child Nc
    "dir_alpha":   0.3,      # Dirichlet concentration for root exploration noise
    "dir_epsilon": 0.25,     # Weight of noise vs network policy at root
}

nn = {
    # abstract_dim: width of the latent state vector (shared across NNr/NNp/NNd).
    # nnr/nnp/nnd_hidden: hidden layer widths — change to adjust depth and width.
    #   e.g. "nnr_hidden": [128, 64]  gives a two-hidden-layer NNr.
    "abstract_dim": 16,
    "nnr_hidden":   [128, 128, 128],
    "nnp_hidden":   [128, 128, 128],
    "nnd_hidden":   [128, 128, 128],    
    "learning_rate": 0.001,
}

training = {
    "num_iterations":    5,
    "episodes_per_iter": 3,
    "epochs_per_iter":   100,
    "roll_ahead":         3,   # w: steps to unroll NNd during BPTT
    # Replay buffer cap. Once full, oldest episode is dropped on each add.
    # Keeps batch size constant → JAX JIT compiles once and reuses every iter.
    # Rule of thumb: episodes_per_iter × ~10 iterations of history.
    "buffer_size":       30,
}

viz = {
    # eval_every: evaluate the model every N training iterations.
    # Each checkpoint plays eval_games MCTS games — set eval_every=0 to skip
    # entirely and avoid slowing down training.
    "eval_every":  0,       # 0 = off; N = evaluate after every N-th iteration
    "eval_games":  2,       # games per checkpoint (small = fast)

    # replay_after_training: render one greedy game to stdout after training.
    # Set False to suppress output when running unattended.
    "replay_after_training": False,
    "replay_max_steps":      1000,

    # compare_baseline: run a random agent after training and overlay its scores
    # on the eval-scores plot. Fast (no network calls) — ~1-2 seconds for 20 games.
    "compare_baseline": True,
    "baseline_games":   10,     # more games = more stable estimate
}
