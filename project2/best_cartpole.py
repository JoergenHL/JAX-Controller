#!/usr/bin/env python3
"""Run N CartPole episodes and save a video of the best one.

Runs all episodes with recording enabled, then keeps only the best video
and deletes the rest. The best episode is the one that survived the most steps.

Usage:
    conda run -n AI python best_cartpole.py                           # latest model, 10 episodes
    conda run -n AI python best_cartpole.py runs/CartPole_<ts>.pkl    # specific model
    conda run -n AI python best_cartpole.py runs/CartPole_<ts>.pkl 20 # 20 episodes
    conda run -n AI python best_cartpole.py runs/CartPole_<ts>.pkl 20 my_video.mp4  # custom output name
"""

import json
import os
import sys
import glob
import shutil
import tempfile

import numpy as np
import jax.numpy as jnp
from flax import nnx

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from nn.NNManager import NNManager
from game.ASM import ASM
import config

_net_fwd = nnx.jit(lambda model, x: model(x))


def find_latest_cartpole(runs_dir: str = "runs") -> str:
    pkls = glob.glob(os.path.join(runs_dir, "CartPole_*.pkl"))
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
        raise ValueError(f"This script is CartPole-only. Model game: {run_data['game']}")

    nn_cfg = run_data.get("config", {}).get("nn", {})
    config.nn["abstract_dim"] = nn_cfg.get("abstract_dim", config.nn["abstract_dim"])
    config.nn["nnr_hidden"]   = nn_cfg.get("nnr_hidden",   config.nn["nnr_hidden"])
    config.nn["nnp_hidden"]   = nn_cfg.get("nnp_hidden",   config.nn["nnp_hidden"])
    config.nn["nnd_hidden"]   = nn_cfg.get("nnd_hidden",   config.nn["nnd_hidden"])

    nnm = NNManager()
    nnm.load(pkl_path)
    return nnm, run_data


def run_episodes(nnm: NNManager, run_data: dict, num_episodes: int,
                 video_dir: str) -> list[dict]:
    """Run num_episodes with recording. Returns list of {episode, steps, video_path}."""
    nn_r = nnm.get_net("nnr")
    nn_p = nnm.get_net("nnp")
    q    = run_data.get("config", {}).get("nn", {}).get("q", 0)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda e: True,   # record every episode
        disable_logger=True,
    )

    results = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        steps  = 0
        done   = False
        state_history = []

        while not done and steps < 500:
            state_history.append(np.array(obs, dtype=np.float32))
            nnr_input = ASM.build_state_window(state_history, q)
            sigma  = _net_fwd(nn_r, jnp.atleast_2d(jnp.array(nnr_input, dtype=jnp.float32)))
            output = _net_fwd(nn_p, sigma)[0]
            action = int(np.argmax(np.array(output[1:])))
            obs, _, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            steps += 1

        video_path = os.path.join(video_dir, f"rl-video-episode-{ep}.mp4")
        results.append({"episode": ep, "steps": steps, "video_path": video_path})

        bar = "#" * (steps // 10) + "-" * ((500 - steps) // 10)
        print(f"  Episode {ep+1:3d}/{num_episodes}: {steps:3d} steps  [{bar}]")

    env.close()
    return results


def save_best(results: list[dict], output_path: str) -> dict:
    """Copy the best episode's video to output_path, delete the rest."""
    best = max(results, key=lambda r: r["steps"])

    if not os.path.exists(best["video_path"]):
        raise FileNotFoundError(
            f"Expected video not found: {best['video_path']}\n"
            "The episode may have been too short for moviepy to flush it."
        )

    shutil.copy2(best["video_path"], output_path)
    return best


def main():
    pkl_path    = sys.argv[1] if len(sys.argv) > 1 else None
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    output_path  = sys.argv[3] if len(sys.argv) > 3 else None

    if pkl_path is None:
        pkl_path = find_latest_cartpole()
        print(f"Auto-selected: {pkl_path}")

    if not os.path.exists(pkl_path):
        print(f"Error: file not found: {pkl_path}")
        sys.exit(1)

    if output_path is None:
        base = os.path.splitext(os.path.basename(pkl_path))[0]
        output_path = f"runs/{base}_best.mp4"

    nnm, run_data = load_model(pkl_path)
    print(f"Loaded:  {os.path.basename(pkl_path)}")
    print(f"Trained: {run_data.get('timestamp', '?')}  "
          f"({len(run_data.get('iterations', []))} iters)")
    print(f"Running {num_episodes} episodes, saving best to: {output_path}\n")

    with tempfile.TemporaryDirectory() as video_dir:
        results = run_episodes(nnm, run_data, num_episodes, video_dir)
        best = save_best(results, output_path)

    scores = [r["steps"] for r in results]
    print(f"\n  avg={sum(scores)/len(scores):.1f}  "
          f"max={max(scores)}  min={min(scores)}")
    print(f"\n  Best episode: #{best['episode']+1} with {best['steps']} steps")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
