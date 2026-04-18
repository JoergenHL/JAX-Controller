"""Worker functions for parallel episode collection and evaluation.

These MUST be module-level functions (not methods or closures) so that
ProcessPoolExecutor can pickle them for subprocess dispatch.

Each worker process:
  1. Receives a frozen snapshot of the network weights as numpy arrays
  2. Reconstructs a local NNManager from scratch (no shared state)
  3. Plays one or more games and returns plain-Python results

Workers use 'spawn' start method, meaning each process imports Python from
scratch and initialises its own JAX instance — no shared mutable state.
"""

import numpy as np


# ── Game registry ──────────────────────────────────────────────────────────────
# Add new game classes here when introducing them.
_GAME_REGISTRY = None

def _get_game(name: str):
    global _GAME_REGISTRY
    if _GAME_REGISTRY is None:
        from game.TwentyFortyEight import TwentyFortyEight
        from game.LineWorld import LineWorld
        from game.CartPole import CartPole
        _GAME_REGISTRY = {
            "TwentyFortyEight": TwentyFortyEight,
            "LineWorld":        LineWorld,
            "CartPole":         CartPole,
        }
    if name not in _GAME_REGISTRY:
        raise ValueError(
            f"Unknown game '{name}'. Add it to _GAME_REGISTRY in worker.py."
        )
    return _GAME_REGISTRY[name]()


# ── Worker entry point ─────────────────────────────────────────────────────────

def collect_episode_worker(args: dict) -> dict:
    """Collect one episode in a worker process.

    This is the top-level callable submitted to ProcessPoolExecutor.
    It fully recreates the game + networks + MCTS from the provided snapshot
    and runs the same episode-collection logic as rlm.collect_episode().

    Args dict keys:
        game_name:     game class name string, e.g. "TwentyFortyEight"
        network_dims:  {name: [dim_in, hidden..., dim_out]} for each network
        layer_weights: output of NNManager.get_layer_weights()
                       {name: [(w_np, b_np), ...]}
        mcts_cfg:      copy of config.mcts dict
        max_steps:     episode step cap (typically 500)

    Returns:
        Episode dict: {states, actions, rewards, policies, returns}
        All entries are plain Python lists / numpy arrays — fully picklable.
    """
    from nn.NNManager import NNManager
    from mcts.mcts import MCTS

    # ── Reconstruct game ───────────────────────────────────────────────────────
    game = _get_game(args["game_name"])

    # ── Reconstruct NNManager from weight snapshot ─────────────────────────────
    nnm = NNManager()
    for name, dims in args["network_dims"].items():
        nnm.create_net(name, dims)
    nnm.set_layer_weights(args["layer_weights"])

    # ── Build MCTS ─────────────────────────────────────────────────────────────
    cfg  = args["mcts_cfg"]
    mcts = MCTS(
        nn_r         = nnm.get_net("nnr"),
        nn_d         = nnm.get_net("nnd"),
        nn_p         = nnm.get_net("nnp"),
        action_space = game.action_space,
        use_puct     = True,
        dir_alpha    = cfg["dir_alpha"],
        dir_epsilon  = cfg["dir_epsilon"],
    )
    mcts.num_simulations = cfg["num_simulations"]
    mcts.c               = cfg["c"]
    mcts.d_max           = cfg["d_max"]

    # ── Episode loop (mirrors rlm.collect_episode) ─────────────────────────────
    from game.ASM import ASM
    from rlm import compute_sampling_tau, temperature_sample
    tau = compute_sampling_tau(
        args.get("iteration", 0),
        args.get("total_iterations", 1),
        args.get("sampling_temp_cfg", {}),
    )

    states, actions, rewards, policies, mcts_values = [], [], [], [], []
    state     = game.initial_state()
    max_steps = args["max_steps"]
    q         = args.get("q", 0)
    steps     = 0
    state_history = []

    while not game.is_terminal(state) and steps < max_steps:
        states.append(np.asarray(state, dtype=np.float32))
        state_history.append(state)
        steps += 1

        nnr_input = ASM.build_state_window(state_history, q)
        _, policy, mcts_val = mcts.search(nnr_input)
        mcts_values.append(float(mcts_val))

        # Sample action from visit-count distribution under temperature τ.
        # Restrict to legal actions so the agent never wastes a step on a move
        # that doesn't change the board (e.g. UP when the board is packed).
        legal = set(game.legal_actions(state))
        legal_policy = {a: v for a, v in policy.items() if a in legal}
        if not legal_policy:
            legal_policy = policy   # fallback: shouldn't happen (is_terminal guard above)
        action = temperature_sample(legal_policy, tau)

        actions.append(action)
        policies.append(policy)

        next_state = game.next_state(state, action)
        reward     = game.reward(state, action, next_state)
        rewards.append(float(reward))

        state = next_state

    return {
        "states":   states,
        "actions":  actions,
        "rewards":  rewards,
        "policies": policies,
        "values":   mcts_values,
    }


def evaluate_greedy_worker(args: dict) -> dict:
    """Play num_games with greedy NNr+NNp in a worker process.

    No MCTS, no NNd — matches the deployed agent in run_agent.py exactly.
    Each game runs to terminal or max_steps, picking argmax of NNp output at
    every step.

    Args dict keys:
        game_name:     game class name string, e.g. "TwentyFortyEight"
        network_dims:  {name: [dim_in, hidden..., dim_out]} for nnr and nnp
        layer_weights: output of NNManager.get_layer_weights() for nnr and nnp
        num_games:     how many games to play
        max_steps:     per-game step cap

    Returns:
        {"wins": int, "max_tiles": [int, ...]}
    """
    import jax.numpy as jnp
    from flax import nnx
    from nn.NNManager import NNManager
    _net_fwd = nnx.jit(lambda model, x: model(x))

    game = _get_game(args["game_name"])

    nnm = NNManager()
    for name, dims in args["network_dims"].items():
        nnm.create_net(name, dims)
    nnm.set_layer_weights(args["layer_weights"])

    nn_r = nnm.get_net("nnr")
    nn_p = nnm.get_net("nnp")
    action_space = game.action_space

    from game.ASM import ASM
    q      = args.get("q", 0)
    wins   = 0
    scores = []

    for _ in range(args["num_games"]):
        state    = game.initial_state()
        max_steps = args["max_steps"]
        steps    = 0
        state_history = []

        while not game.is_terminal(state) and steps < max_steps:
            state_history.append(state)
            nnr_input = ASM.build_state_window(state_history, q)
            sigma  = _net_fwd(nn_r, jnp.atleast_2d(
                jnp.array(nnr_input, dtype=jnp.float32)
            ))
            output = _net_fwd(nn_p, sigma)[0]
            # output[0] = value; output[1:] = action logits.
            # Mask illegal actions to -inf so argmax always picks a legal move.
            legal = game.legal_actions(state)
            logits = output[1:]
            masked = [float(logits[i]) if action_space[i] in legal else float('-inf')
                      for i in range(len(action_space))]
            action = action_space[int(np.argmax(masked))]
            state  = game.next_state(state, action)
            steps += 1

        if game.is_win(state):
            wins += 1
        scores.append(game.eval_score(steps, state))

    return {"wins": wins, "scores": scores}
