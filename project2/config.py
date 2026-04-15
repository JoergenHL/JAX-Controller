# Central configuration for the MuZero project.
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
    "num_simulations": 50,   # Simulations per move
    "c": 2,                  # Exploration constant in PUCT formula
    "d_max": 1,              # Rollout depth: NNd steps from the randomly-picked child Nc
    "dir_alpha":   0.3,      # Dirichlet concentration for root exploration noise
    "dir_epsilon": 0.25,     # Weight of noise vs network policy at root
}

nn = {
    # abstract_dim: width of the latent state vector (shared across NNr/NNp/NNd).
    # nnr/nnp/nnd_hidden: hidden layer widths — change to adjust depth and width.
    #   e.g. "nnr_hidden": [128, 64]  gives a two-hidden-layer NNr.
    "abstract_dim": 32,
    "nnr_hidden":   [128, 128, 128],
    "nnp_hidden":   [128, 128, 128],
    "nnd_hidden":   [128, 128, 128],    
    "learning_rate": 0.001,
    # MuZero paper weights value loss at 0.25 — prevents it being crowded out
    # by policy cross-entropy, which occupies a larger scale early in training.
    "loss_weights": {"value": 0.25, "policy": 1.0, "reward": 1.0},
}

training = {
    "num_iterations":    50,
    "episodes_per_iter": 9,
    "epochs_per_iter":   100,
    "roll_ahead":         3,   # w: steps to unroll NNd during BPTT
    # Replay buffer cap. Once full, oldest episode is dropped on each add.
    # Keeps batch size constant → JAX JIT compiles once and reuses every iter.
    # Rule of thumb: episodes_per_iter × ~10 iterations of history.
    "buffer_size":       90 ,
    # N-step bootstrap horizon (MuZero Section 3).
    # Value target at step k: z_k = r_{k+1}+...+r_{k+n} + v_{k+n}
    # Uses MCTS value estimate v_{k+n} to cap variance at n steps instead
    # of summing 200-400 stochastic rewards to end of game.
    "n_step":            10,
    # Discount factor for Monte Carlo returns.
    # γ=0.99 means a reward 200 steps away is worth 0.99^200 ≈ 0.13 now.
    # This reduces variance from distant stochastic tile spawns without
    # the feedback-loop instability of n-step bootstrap.
    "gamma":             0.99,
    # Parallel episode collectors. Each worker is a separate process with its
    # own JAX instance and a copy of the current network weights.
    # Set to 1 to disable parallelism (sequential, easier to debug).
    # Maximum useful value = episodes_per_iter (one worker per episode).
    "num_workers":        3,
}

viz = {
    # eval_every: evaluate the model every N training iterations.
    # Each checkpoint plays eval_games greedy NNr+NNp games — fast, matches
    # deployment. Set eval_every=0 to skip entirely.
    "eval_every":  1,       # 0 = off; N = evaluate after every N-th iteration
    "eval_games":  5,       # games per checkpoint

    # replay_after_training: render one greedy game to stdout after training.
    # Set False to suppress output when running unattended.
    "replay_after_training": False,
    "replay_max_steps":      50,

    # compare_baseline: run a random agent after training and overlay its scores
    # on the eval-scores plot. Fast (no network calls) — ~1-2 seconds for 20 games.
    "compare_baseline": True,
    "baseline_games":   100,     # more games = more stable estimate

    # compare_mcts: run full MCTS eval (NNr+NNd+NNp) once after training.
    # Shows how much planning adds on top of the greedy NNr+NNp policy.
    # Slow (100 sims/step) — keep mcts_eval_games small.
    "compare_mcts":     False,
    "mcts_eval_games":  10,

    # show_policy_analysis: after training, sample greedy games and plot three
    # diagnostics of the learned policy (no MCTS, <1s).
    #   Panel 1 — action preference bar chart (which directions are chosen most?)
    #   Panel 2 — policy entropy histogram (confident policy vs. uniform/untrained)
    #   Panel 3 — value estimate vs. board quality scatter (has value learned?)
    # Saved as <run_name>_policy.png alongside the training plot.
    "show_policy_analysis":  True,
    "policy_analysis_games": 20,   # greedy games to sample (~200 steps each)
}
