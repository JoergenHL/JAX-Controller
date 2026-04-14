#!/usr/bin/env python3
"""Load a trained agent and watch it play.

Runs NNr → NNp directly (no MCTS or NNd) — the deployment model for a
trained MuZero agent. NNr encodes the real board state to abstract space;
NNp maps that to a policy distribution; argmax picks the action.

Usage:
    python run_agent.py                                 # auto-load latest run
    python run_agent.py runs/<GameClass>_<ts>.pkl       # specific model
    python run_agent.py runs/<GameClass>_<ts>.pkl 200   # custom step limit
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
from flax import nnx

from game.LineWorld import LineWorld
from game.TwentyFortyEight import TwentyFortyEight
from nn.NNManager import NNManager
import config

# ── Game registry ──────────────────────────────────────────────────────────────
# Add new games here when they are introduced.
GAMES = {
    "TwentyFortyEight": TwentyFortyEight,
    "LineWorld":        LineWorld,
}

_net_fwd = nnx.jit(lambda model, x: model(x))


def find_latest_model(runs_dir: str = "runs") -> str:
    """Return the path of the most recently modified .pkl in runs/."""
    pkls = [
        os.path.join(runs_dir, f)
        for f in os.listdir(runs_dir)
        if f.endswith(".pkl")
    ]
    if not pkls:
        raise FileNotFoundError(f"No .pkl files found in '{runs_dir}/'")
    return max(pkls, key=os.path.getmtime)


def load_model(pkl_path: str):
    """Load NNManager from pkl and game from companion JSON.

    Returns:
        game:    instantiated game object
        nnm:     NNManager with nnr and nnp loaded
        run_data: the full JSON dict for reference
    """
    json_path = pkl_path.replace(".pkl", ".json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Companion JSON not found: {json_path}\n"
            "Make sure the .pkl and .json share the same basename."
        )

    with open(json_path) as f:
        run_data = json.load(f)

    game_name = run_data["game"]
    if game_name not in GAMES:
        raise ValueError(
            f"Unknown game '{game_name}'. "
            f"Add it to the GAMES registry in run_agent.py."
        )
    game = GAMES[game_name]()

    nnm = NNManager()
    nnm.load(pkl_path)

    return game, nnm, run_data


def play(game, nnm: NNManager, max_steps: int = None):
    """Play one greedy episode using NNr → NNp, no MCTS.

    NNr encodes the real state to abstract space once per step.
    NNp maps abstract state → (value, policy logits).
    Action = argmax of policy logits (greedy, deterministic).

    Args:
        game:      game instance
        nnm:       NNManager with 'nnr' and 'nnp' loaded
        max_steps: episode length limit (falls back to config value)
    """
    if max_steps is None:
        max_steps = config.viz.get("replay_max_steps", 400)

    nn_r = nnm.get_net("nnr")
    nn_p = nnm.get_net("nnp")
    action_space = game.action_space

    state        = game.initial_state()
    total_reward = 0.0
    steps        = 0

    sep = "─" * 44
    print(f"\n{sep}")
    print(f"  Greedy replay  (NNr + NNp, max {max_steps} steps)")
    print(f"  Game: {game.__class__.__name__}")
    print(sep)
    print("Initial board:")
    game.render(state)

    while not game.is_terminal(state) and steps < max_steps:
        # NNr: real state → abstract state σ
        sigma = _net_fwd(nn_r, jnp.atleast_2d(
            jnp.array(state, dtype=jnp.float32)
        ))

        # NNp: σ → (value, policy logits)
        output = _net_fwd(nn_p, sigma)[0]    # shape [1 + num_actions]
        # output[0] = value, output[1:] = policy logits
        action_idx = int(jnp.argmax(output[1:]))
        action     = action_space[action_idx]

        next_state   = game.next_state(state, action)
        reward       = game.reward(state, action, next_state)
        total_reward += reward
        steps        += 1

        print(f"Step {steps:3d} | action={action:<6} | reward={reward:6.1f} "
              f"| max tile={game.max_tile(next_state)}")
        game.render(next_state)

        state = next_state

    print(sep)
    print(f"  Episode ended  steps={steps}  total_reward={total_reward:.1f}"
          f"  max_tile={game.max_tile(state)}")
    print(f"{sep}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Parse arguments
    pkl_path  = sys.argv[1] if len(sys.argv) > 1 else None
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if pkl_path is None:
        pkl_path = find_latest_model()
        print(f"Auto-selected: {pkl_path}")

    if not os.path.exists(pkl_path):
        print(f"Error: file not found: {pkl_path}")
        sys.exit(1)

    game, nnm, run_data = load_model(pkl_path)

    print(f"\nLoaded run: {os.path.basename(pkl_path)}")
    print(f"  Game:        {run_data['game']}")
    print(f"  Trained:     {run_data.get('timestamp', '?')}")
    print(f"  Iterations:  {len(run_data.get('iterations', []))}")

    play(game, nnm, max_steps=max_steps)
