#!/usr/bin/env python3
"""Render the 2048 champion playing one full game as an MP4.

What the output shows (runs/<prefix>_viz/play.mp4):
  - Left panel  : the 4x4 board, tiles coloured by the standard 2048 palette
                  (cream for small tiles, orange/red for mid, gold for 256+).
                  Tile value drawn inside each cell. This is the whole game
                  unrolled step-by-step so viewers can watch the agent's
                  strategy emerge (corner-hugging, tile stacking, etc.).
  - Right panel : a live bar chart of the MCTS policy pi = visit counts at
                  the root. The action the agent actually took is highlighted
                  in orange. Tall bars reflect which move the tree search
                  converged on; flat bars mean the search found no clearly
                  better option.
  - Header      : step index, running score (sum of merge values), current
                  max tile.
  - Subtitle    : MCTS root value v and reward gained on this step.

By default the agent plays with full u-MCTS enabled (nnr -> nnd/nnp tree search),
matching what MuZero actually does at inference time. Pass --greedy to skip MCTS
and use NNr + NNp argmax only -- faster to render, less faithful to the paper.

Usage:
  conda run -n AI python viz_2048_play.py                          # latest champion
  conda run -n AI python viz_2048_play.py runs/TwentyFortyEight_<ts>_champion.pkl
  conda run -n AI python viz_2048_play.py <pkl> --greedy
  conda run -n AI python viz_2048_play.py <pkl> --fps 6 --max-steps 1000
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import jax.numpy as jnp
import jax
from flax import nnx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import FancyBboxPatch

from nn.NNManager import NNManager
from game.TwentyFortyEight import TwentyFortyEight
from game.ASM import ASM
from mcts.mcts import MCTS
import config


# Standard 2048 tile palette keyed by log2(tile). 0 = empty cell.
TILE_COLOURS = {
    0:  "#cdc1b4",   # empty
    1:  "#eee4da",   # 2
    2:  "#ede0c8",   # 4
    3:  "#f2b179",   # 8
    4:  "#f59563",   # 16
    5:  "#f67c5f",   # 32
    6:  "#f65e3b",   # 64
    7:  "#edcf72",   # 128
    8:  "#edcc61",   # 256
    9:  "#edc850",   # 512
    10: "#edc53f",   # 1024
    11: "#edc22e",   # 2048
    12: "#3c3a32",   # 4096+
}
TEXT_DARK_ON_LIGHT_THRESHOLD = 2   # tiles <= 4 get dark text; larger tiles get white


_net_fwd = nnx.jit(lambda model, x: model(x))


# ---- Model loading -----------------------------------------------------------

def find_latest_champion(runs_dir: str = "runs") -> str:
    pkls = glob.glob(os.path.join(runs_dir, "TwentyFortyEight_*_champion.pkl"))
    if not pkls:
        raise FileNotFoundError(
            "No TwentyFortyEight_*_champion.pkl files found in 'runs/'")
    return max(pkls, key=os.path.getmtime)


def load_model(pkl_path: str):
    """Load champion weights and patch config from the companion JSON.

    Mirrors best_cartpole.py's loader: the JSON stores the exact hparams that
    trained this model (abstract_dim, q, MCTS settings). We overwrite the
    module-level config so MCTS picks up the right values when instantiated.
    """
    json_path = pkl_path.replace(".pkl", ".json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Companion JSON not found: {json_path}")

    with open(json_path) as f:
        run_data = json.load(f)

    if run_data.get("game") != "TwentyFortyEight":
        raise ValueError(
            f"This script is 2048-only. Model game: {run_data.get('game')}")

    nn_cfg   = run_data.get("config", {}).get("nn",   {})
    mcts_cfg = run_data.get("config", {}).get("mcts", {})

    config.nn["abstract_dim"] = nn_cfg.get("abstract_dim", config.nn["abstract_dim"])
    config.nn["q"]            = nn_cfg.get("q",            config.nn.get("q", 0))
    config.mcts.update({
        "num_simulations": mcts_cfg.get("num_simulations", config.mcts["num_simulations"]),
        "c":               mcts_cfg.get("c",               config.mcts["c"]),
        "d_max":           mcts_cfg.get("d_max",           config.mcts["d_max"]),
        "dir_alpha":       mcts_cfg.get("dir_alpha",       config.mcts["dir_alpha"]),
        "dir_epsilon":     mcts_cfg.get("dir_epsilon",     config.mcts["dir_epsilon"]),
    })

    nnm = NNManager()
    nnm.load(pkl_path)
    return nnm, run_data


# ---- Playing one game --------------------------------------------------------

def play_game(nnm: NNManager, use_mcts: bool, max_steps: int, seed: int = None):
    """Play one 2048 game, returning a list of per-step snapshots for rendering.

    Each snapshot holds everything a frame needs to draw: the board, the
    action taken, the MCTS visit distribution (or a degenerate one-hot from
    the greedy argmax if MCTS is off), the root value, and rewards.
    """
    if seed is not None:
        np.random.seed(seed)
        import random as _r
        _r.seed(seed)

    gsm = TwentyFortyEight()
    q   = config.nn.get("q", 0)

    if use_mcts:
        mcts = MCTS(
            nn_r         = nnm.get_net("nnr"),
            nn_d         = nnm.get_net("nnd"),
            nn_p         = nnm.get_net("nnp"),
            action_space = gsm.action_space,
            use_puct     = True,
            dir_alpha    = config.mcts["dir_alpha"],
            dir_epsilon  = config.mcts["dir_epsilon"],
        )
        mcts.num_simulations = config.mcts["num_simulations"]
        mcts.c               = config.mcts["c"]
        mcts.d_max           = config.mcts["d_max"]
    else:
        mcts = None
        nn_r = nnm.get_net("nnr")
        nn_p = nnm.get_net("nnp")

    action_space = gsm.action_space
    state = gsm.initial_state()
    state_history = []
    frames = []
    score  = 0
    steps  = 0

    while not gsm.is_terminal(state) and steps < max_steps:
        state_history.append(state)
        nnr_input = ASM.build_state_window(state_history, q)

        if use_mcts:
            action, policy, value = mcts.search(nnr_input)
            # Restrict sampling to legal actions (same safeguard as rlm.collect_episode).
            legal = set(gsm.legal_actions(state))
            legal_policy = {a: v for a, v in policy.items() if a in legal}
            if legal_policy and max(legal_policy.values()) > 0:
                max_v   = max(legal_policy.values())
                winners = [a for a, v in legal_policy.items() if v == max_v]
                action  = winners[0]
        else:
            # Greedy NNr -> NNp argmax. Show the softmax probability over all
            # four actions in the bar chart so viewers can see the network's
            # confidence, not just which action won.
            sigma  = _net_fwd(nn_r, jnp.atleast_2d(
                jnp.array(nnr_input, dtype=jnp.float32)))
            output = _net_fwd(nn_p, sigma)[0]
            value  = float(output[0])
            logits = np.array(output[1:])
            legal  = gsm.legal_actions(state)
            mask   = np.array([a in legal for a in action_space])
            masked = np.where(mask, logits, -np.inf)
            action = action_space[int(np.argmax(masked))]
            probs  = np.array(jax.nn.softmax(logits))
            # Scale up so the bar chart uses the same ~0..1 axis as MCTS mode.
            policy = {a: float(probs[i]) for i, a in enumerate(action_space)}

        next_state = gsm.next_state(state, action)
        reward = gsm.reward(state, action, next_state)
        # Score = sum of actual merged tile values (standard 2048 scoring).
        # gsm.reward returns log2 of the biggest merged tile; to display the
        # running score we recompute merge_score directly.
        _, _, merge_score = gsm._apply_move(state, action)
        score += merge_score

        frames.append({
            "board":     state.reshape(4, 4).copy(),
            "step":      steps + 1,
            "action":    action,
            "policy":    dict(policy),
            "value":     float(value),
            "reward":    float(reward),
            "merge":     int(merge_score),
            "score":     int(score),
            "max_tile":  gsm.max_tile(state),
        })

        state = next_state
        steps += 1

    # Final (terminal) frame so the viewer sees the end-of-game board.
    frames.append({
        "board":     state.reshape(4, 4).copy(),
        "step":      steps + 1,
        "action":    None,
        "policy":    {a: 0 for a in action_space},
        "value":     0.0,
        "reward":    0.0,
        "merge":     0,
        "score":     int(score),
        "max_tile":  gsm.max_tile(state),
        "terminal":  True,
    })

    return frames, action_space


# ---- Rendering ---------------------------------------------------------------

def draw_board(ax, board: np.ndarray):
    """Render the 4x4 board on ax. board holds log2(tile) values."""
    ax.clear()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Background panel for the whole grid (matches 2048's beige).
    ax.add_patch(FancyBboxPatch(
        (0, 0), 4, 4,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=0, facecolor="#bbada0",
    ))

    for r in range(4):
        for c in range(4):
            # Flip row so row 0 renders at the top (matches gsm.render() order).
            val_log2 = int(board[r, c])
            colour   = TILE_COLOURS.get(val_log2, TILE_COLOURS[12])
            y = 3 - r

            ax.add_patch(FancyBboxPatch(
                (c + 0.05, y + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=0, facecolor=colour,
            ))
            if val_log2 > 0:
                tile = 2 ** val_log2
                # Dark text on the two palest tiles, white on everything else.
                text_colour = "#776e65" if val_log2 <= TEXT_DARK_ON_LIGHT_THRESHOLD else "white"
                # Shrink the font for 4-digit tiles so they fit inside the cell.
                font_size = 34 if tile < 100 else (28 if tile < 1000 else 22)
                ax.text(c + 0.5, y + 0.5, str(tile),
                        ha="center", va="center",
                        fontsize=font_size, fontweight="bold",
                        color=text_colour)


def draw_policy(ax, policy: dict, action_space: list, chosen_action: str,
                mode: str = "mcts"):
    """Bar chart of the policy over the 4 actions. Chosen action in orange.

    mode="mcts"   : policy values are raw visit counts; bars show visit share.
    mode="greedy" : policy values are NNp softmax probabilities; bars show
                    those probabilities directly.
    """
    ax.clear()
    counts = [policy.get(a, 0) for a in action_space]
    total  = sum(counts) or 1
    probs  = [c / total for c in counts]

    # Grey for non-chosen, orange for the action actually executed this step.
    colours = ["#f59563" if a == chosen_action else "#bbada0"
               for a in action_space]

    bars = ax.bar(range(len(action_space)), probs, color=colours,
                  edgecolor="#776e65", linewidth=1.2)
    ax.set_xticks(range(len(action_space)))
    ax.set_xticklabels(action_space, fontsize=11, fontweight="bold",
                       color="#776e65")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    if mode == "mcts":
        ax.set_ylabel("MCTS visit share pi", fontsize=10, color="#776e65")
        ax.set_title("Search policy at this step",
                     fontsize=11, color="#776e65", pad=8)
    else:
        ax.set_ylabel("NNp softmax probability", fontsize=10, color="#776e65")
        ax.set_title("Policy network at this step",
                     fontsize=11, color="#776e65", pad=8)
    ax.tick_params(colors="#776e65")
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color("#bbada0")

    # Label each bar with its raw MCTS visit count (integers) or NNp
    # probability (0..1) depending on what the caller passed in.
    for bar, c in zip(bars, counts):
        label = f"{int(c)}" if isinstance(c, (int, np.integer)) else f"{c:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                label, ha="center", va="bottom",
                fontsize=9, color="#776e65")


def render_video(frames: list, action_space: list, out_path: str,
                 fps: int = 4, title_suffix: str = "",
                 policy_mode: str = "mcts") -> str:
    """Write all frames to an MP4 (FFMpegWriter) or GIF fallback."""
    fig, (ax_board, ax_pol) = plt.subplots(
        1, 2, figsize=(10, 6),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )
    fig.patch.set_facecolor("#faf8ef")   # 2048 background
    # Reserve headroom so the top-bar header/subtitle never collides with the
    # right-panel title (matplotlib places subplot titles *above* the axes).
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.04, right=0.98,
                        wspace=0.22)

    # Persistent text handles above the figure so we only update their strings.
    header   = fig.text(0.5, 0.955, "", ha="center", va="top",
                        fontsize=15, fontweight="bold", color="#776e65")
    subtitle = fig.text(0.5, 0.900, "", ha="center", va="top",
                        fontsize=10, color="#776e65")

    def update(i):
        f = frames[i]
        draw_board(ax_board, f["board"])
        draw_policy(ax_pol, f["policy"], action_space, f.get("action"),
                    mode=policy_mode)

        extra = ""
        if f.get("terminal"):
            extra = "  -  GAME OVER"
        header.set_text(
            f"Step {f['step']:3d}   score {f['score']:5d}   "
            f"max tile {f['max_tile']:5d}{extra}"
        )
        act_str = f["action"] if f["action"] else "-"
        reward_str = (f"merge +{f['merge']}" if f["merge"] > 0 else "no merge")
        subtitle.set_text(
            f"action {act_str:<5}   root value v = {f['value']:+.2f}   "
            f"{reward_str}{title_suffix}"
        )
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps,
                         blit=False, repeat=False)

    # Try MP4 first; fall back to GIF if ffmpeg isn't usable for some reason.
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(out_path, writer=writer, dpi=140)
    except Exception as e:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        print(f"  FFMpeg failed ({e}). Falling back to GIF: {gif_path}")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
        out_path = gif_path

    plt.close(fig)
    return out_path


# ---- CLI ---------------------------------------------------------------------

def derive_viz_dir(pkl_path: str) -> str:
    """runs/FOO_champion.pkl -> runs/FOO_viz/  (also strips _best1..5)."""
    prefix = os.path.splitext(pkl_path)[0]
    for suffix in ("_champion", "_best1", "_best2", "_best3", "_best4", "_best5"):
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break
    return f"{prefix}_viz"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pkl_path", nargs="?", default=None,
                        help="Champion .pkl (default: latest in runs/)")
    parser.add_argument("--greedy", action="store_true",
                        help="Play with NNr+NNp only (no MCTS); faster")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Safety cap on game length")
    parser.add_argument("--fps", type=int, default=4,
                        help="Output framerate (default 4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for reproducible spawns / Dirichlet noise")
    parser.add_argument("--out", default=None,
                        help="Output path (default: runs/<prefix>_viz/play.mp4)")
    args = parser.parse_args()

    pkl_path = args.pkl_path or find_latest_champion()
    if not os.path.exists(pkl_path):
        print(f"Error: file not found: {pkl_path}", file=sys.stderr)
        sys.exit(1)

    nnm, run_data = load_model(pkl_path)
    print(f"Loaded:    {os.path.basename(pkl_path)}")
    print(f"Trained:   {run_data.get('timestamp', '?')}")
    print(f"Eval avg:  {run_data.get('final_eval_avg', run_data.get('eval_avg', '?'))} "
          f"(from training metadata)")
    print(f"MCTS:      sims={config.mcts['num_simulations']}  "
          f"c={config.mcts['c']:.3f}  d_max={config.mcts['d_max']}  "
          f"dir_eps={config.mcts['dir_epsilon']:.3f}")
    print(f"NN:        abstract_dim={config.nn['abstract_dim']}  q={config.nn.get('q', 0)}")

    mode = "greedy (no MCTS)" if args.greedy else "u-MCTS"
    print(f"\nPlaying 1 game in {mode} mode...")
    frames, action_space = play_game(
        nnm, use_mcts=not args.greedy,
        max_steps=args.max_steps, seed=args.seed,
    )
    last = frames[-1]
    print(f"  Episode ended at step {last['step']}: "
          f"score={last['score']}  max_tile={last['max_tile']}")

    viz_dir = derive_viz_dir(pkl_path)
    os.makedirs(viz_dir, exist_ok=True)
    out_path = args.out or os.path.join(viz_dir, "play.mp4")

    print(f"\nRendering {len(frames)} frames at {args.fps} fps...")
    final_path = render_video(
        frames, action_space, out_path, fps=args.fps,
        title_suffix=("  -  " + mode),
        policy_mode=("mcts" if not args.greedy else "greedy"),
    )
    abs_path = os.path.abspath(final_path)
    print(f"\n  Saved -> {abs_path}")


if __name__ == "__main__":
    main()
