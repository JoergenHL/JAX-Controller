# Central configuration for the MuZero project.
# All hyperparameters live here so they can be changed in one place.
#
# Network dimensions are NOT stored here — they are derived at startup from
# the game object (game.state_dim, game.num_actions) and the tuning knobs below.
#
#   NNr full dims: [state_dim * (q+1)]       + nnr_hidden + [abstract_dim]
#   NNp full dims: [abstract_dim]           + nnp_hidden + [1 + num_actions]
#   NNd full dims: [abstract_dim+num_actions] + nnd_hidden + [abstract_dim + 1]
#
# To change network depth/width, edit the hidden lists below (e.g. [128, 64]).
# To swap games, change game["name"] below — no other file needs to be touched.

# ── Game selection ─────────────────────────────────────────────────────────────
# name: class name string — must match an entry in worker.py's _GAME_REGISTRY.
# Available: "TwentyFortyEight"
game = {
    "name": "TwentyFortyEight",
}

mcts = {
    "num_simulations": 50,   # Simulations per move
    "c": 0.4,                  # Exploration constant in PUCT formula
    "d_max": 2,              # Rollout depth: NNd steps from the randomly-picked child Nc
    "dir_alpha":   0.3,      # Dirichlet concentration for root exploration noise
    "dir_epsilon": 0.23,     # Weight of noise vs network policy at root
}

nn = {
    # abstract_dim: width of the latent state vector (shared across NNr/NNp/NNd).
    # nnr/nnp/nnd_hidden: hidden layer widths — change to adjust depth and width.
    #   e.g. "nnr_hidden": [128, 64]  gives a two-hidden-layer NNr.
    # q: look-back window for NNr. NNr receives the last q+1 real states concatenated.
    #   NNr input dim = state_dim * (q+1).
    #   q=0: single state (original behaviour).
    #   q=2: three consecutive states — gives NNr access to velocity and acceleration.
    "abstract_dim": 32,
    "q":            2,
    "nnr_hidden":   [128, 128],
    "nnp_hidden":   [128, 128],
    "nnd_hidden":   [128, 128],    
    "learning_rate": 0.001,
    # LR schedule across all gradient steps of a training run.
    #   "constant": flat `learning_rate` for the entire run (original behaviour).
    #   "cosine":   cosine decay from `learning_rate` → `learning_rate * lr_floor_frac`
    #               over num_iterations * updates_per_iter steps. Smoother late
    #               training, less oscillation, cleaner leaderboard convergence.
    "lr_schedule":   "cosine",
    "lr_floor_frac": 0.1,
    # MuZero paper weights value loss at 0.25 — prevents it being crowded out
    # by policy cross-entropy, which occupies a larger scale early in training.
    "loss_weights": {"value": 0.25, "policy": 1.0, "reward": 1.0},
}

training = {
    "num_iterations":    10,
    "episodes_per_iter": 9,
    # MuZero-style minibatching: many small random-sample gradient steps per
    # iteration instead of full-batch epoch passes. Each update samples
    # `minibatch_size` random (episode, step) windows from the buffer.
    # Tuning rationale: with ~13k windows in the buffer and mbs=128,
    # 500 updates exposes each window ~5× on average per iteration — enough
    # signal to track the moving target without memorising any snapshot.
    "updates_per_iter":  50,
    "minibatch_size":    64,
    "roll_ahead":         3,   # w: steps to unroll NNd during BPTT
    # Replay buffer history depth (in training iterations). The actual capacity
    # is derived at RLM init as:
    #   episodes_per_iter * min(buffer_history_iters, num_iterations)
    # so the buffer always sizes itself to the current config without needing
    # a raw episode count. Set ~100 for long overnight runs; the min() clamp
    # keeps short hparam searches from over-allocating.
    # Callers that want an exact raw cap may still set training["buffer_size"]
    # directly — that path wins when present.
    "buffer_history_iters": 100,
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
    # Self-play action-sampling temperature schedule. At τ=1, actions are
    # sampled proportional to raw MCTS visit counts (exploration-friendly).
    # At τ→0, sampling concentrates on the most-visited action (exploitation).
    # Schedule:
    #   iteration < transition_at * num_iterations  → τ = start
    #   otherwise → linearly interp start → end across the remaining iterations
    # Sampling rule: p ∝ visit_count^(1/τ), then renormalise.
    "sampling_temp": {
        "start":         1.0,
        "end":           0.2,
        "transition_at": 0.5,   # fraction of training before decay begins
    },
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
    "eval_games":  50,      # games per checkpoint (used for leaderboard selection)

    # Interval snapshotting for video progression — pkl + companion JSON saved
    # every N iterations so we can later play the agent at different points in
    # training. 0 disables entirely (hparam searches should keep it off).
    "checkpoint_every": 20,

    # Top-K "best model" leaderboard. After each iteration's eval, if the
    # eval avg exceeds best_threshold AND beats the worst entry, the model
    # enters the leaderboard. After training, every entry is re-evaluated on
    # final_eval_games games for an authoritative ranking; the winner becomes
    # <prefix>_champion.pkl. Set best_leaderboard_k=0 to disable.
    "best_leaderboard_k": 5,
    "best_threshold":     50,    # min eval avg required to enter leaderboard
    "final_eval_games":   1000,  # games per candidate in the post-training shootout

    # replay_after_training: render one greedy game to stdout after training.
    # Set False to suppress output when running unattended.
    "replay_after_training": False,
    "replay_max_steps":      500,

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
    "policy_analysis_games": 100,   # greedy games to sample (~200 steps each)
}
