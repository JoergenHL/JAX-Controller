#!/usr/bin/env python3
"""Watch a trained CartPole agent play using the gym renderer.

Loads a saved model (NNr + NNp) and drives the real gymnasium CartPole-v1
environment so you get the graphical window. The agent policy is identical
to evaluation: NNr encodes the gym observation → NNp gives policy logits →
argmax picks the action. No MCTS.

Usage:
    conda run -n AI python watch_cartpole.py                         # latest run
    conda run -n AI python watch_cartpole.py runs/CartPole_<ts>.pkl  # specific
    conda run -n AI python watch_cartpole.py runs/CartPole_<ts>.pkl 3  # 3 games
"""

import json
import os
import sys

import numpy as np
import jax.numpy as jnp
from flax import nnx

import gymnasium as gym

from nn.NNManager import NNManager
from game.ASM import ASM
import config

_net_fwd = nnx.jit(lambda model, x: model(x))

ACTIONS = ["LEFT", "RIGHT"]   # index 0 = LEFT = gym action 0, RIGHT = 1


def find_latest_cartpole(runs_dir: str = "runs") -> str:
    pkls = [
        os.path.join(runs_dir, f)
        for f in os.listdir(runs_dir)
        if f.endswith(".pkl") and "CartPole" in f
    ]
    if not pkls:
        raise FileNotFoundError("No CartPole .pkl files found in 'runs/'")
    return max(pkls, key=os.path.getmtime)


def load_model(pkl_path: str):
    json_path = pkl_path.replace(".pkl", ".json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Companion JSON not found: {json_path}")

    with open(json_path) as f:
        run_data = json.load(f)

    if run_data.get("game") != "CartPole":
        raise ValueError(
            f"This script is CartPole-only. Model is for: {run_data['game']}"
        )

    # Restore network dimensions from the saved run so NNManager builds
    # networks with the right shapes before loading weights.
    nn_cfg = run_data.get("config", {}).get("nn", {})
    config.nn["abstract_dim"] = nn_cfg.get("abstract_dim", config.nn["abstract_dim"])
    config.nn["nnr_hidden"]   = nn_cfg.get("nnr_hidden",   config.nn["nnr_hidden"])
    config.nn["nnp_hidden"]   = nn_cfg.get("nnp_hidden",   config.nn["nnp_hidden"])
    config.nn["nnd_hidden"]   = nn_cfg.get("nnd_hidden",   config.nn["nnd_hidden"])

    nnm = NNManager()
    nnm.load(pkl_path)

    return nnm, run_data


def watch(nnm: NNManager, run_data: dict, num_games: int = 1, render_fps: int = 30):
    """Play num_games episodes with the gym renderer open."""
    nn_r = nnm.get_net("nnr")
    nn_p = nnm.get_net("nnp")
    q    = run_data.get("config", {}).get("nn", {}).get("q", 0)

    env = gym.make("CartPole-v1", render_mode="human")
    env.metadata["render_fps"] = render_fps

    scores = []
    for game_idx in range(1, num_games + 1):
        obs, _ = env.reset()
        steps  = 0
        done   = False
        state_history = []

        while not done and steps < 500:
            state_history.append(np.array(obs, dtype=np.float32))
            nnr_input = ASM.build_state_window(state_history, q)
            sigma = _net_fwd(
                nn_r,
                jnp.atleast_2d(jnp.array(nnr_input, dtype=jnp.float32))
            )
            # NNp: abstract state → (value, policy logits)
            output  = _net_fwd(nn_p, sigma)[0]   # shape [1 + num_actions]
            logits  = output[1:]
            action  = int(np.argmax(np.array(logits)))   # 0=LEFT, 1=RIGHT

            obs, _, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            steps += 1

        scores.append(steps)
        print(f"  Game {game_idx:2d}: {steps} steps")

    env.close()
    print(f"\n  avg={sum(scores)/len(scores):.1f}  "
          f"max={max(scores)}  min={min(scores)}")


if __name__ == "__main__":
    pkl_path  = sys.argv[1] if len(sys.argv) > 1 else None
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if pkl_path is None:
        pkl_path = find_latest_cartpole()
        print(f"Auto-selected: {pkl_path}")

    if not os.path.exists(pkl_path):
        print(f"Error: file not found: {pkl_path}")
        sys.exit(1)

    nnm, run_data = load_model(pkl_path)
    print(f"Loaded: {os.path.basename(pkl_path)}")
    print(f"  Trained: {run_data.get('timestamp', '?')}")
    print(f"  Iters:   {len(run_data.get('iterations', []))}")
    print(f"  Playing {num_games} game(s)...\n")

    watch(nnm, run_data, num_games=num_games)
