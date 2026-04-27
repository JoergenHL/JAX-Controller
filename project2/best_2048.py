#!/usr/bin/env python3
"""Play N games of 2048 with the champion and render the best one as an MP4.

Plays N full games (greedy NNr+NNp by default -- matches training evaluation),
ranks them by max tile (tie-break by game score, then length), and renders
only the winning game using the same V1 pipeline as viz_2048_play.py. All
other games are discarded so disk usage stays tiny.

What the output shows (runs/<prefix>_viz/best_of_N.mp4):
  - the best single game out of N attempts, rendered step-by-step with the
    2048 board on the left and the policy-net probabilities on the right.
  - See viz_2048_play.py's module docstring for the panel-by-panel breakdown.

Why "best of N" is a sensible demo: one greedy 2048 game is high-variance
(spawn tiles are random). The mean max tile is ~193 but individual games
regularly score 64 or 512. Picking the best of N gives a representative
*upper-tail* game without cherry-picking a seed by hand.

Usage:
  conda run -n AI python best_2048.py                          # latest champion, 50 games
  conda run -n AI python best_2048.py <pkl>                    # 50 games
  conda run -n AI python best_2048.py <pkl> 100                # 100 games
  conda run -n AI python best_2048.py <pkl> 100 --mcts         # use u-MCTS (slower)
  conda run -n AI python best_2048.py <pkl> 100 --out custom.mp4
"""

import argparse
import os
import sys

import numpy as np

# Import V1 primitives so there's a single source of truth for how a 2048
# game is played and rendered -- this script is just a "play N games + render
# the best one" wrapper around those.
from viz_2048_play import (
    find_latest_champion,
    load_model,
    play_game,
    render_video,
    derive_viz_dir,
)
import config


def score_game(frames: list) -> tuple:
    """Rank key for a completed game -- higher is better.

    Primary: max tile reached (matches training's eval_score for 2048).
    Tie-break 1: total game score (sum of merge values).
    Tie-break 2: number of steps survived (longer usually means more merges).
    """
    last = frames[-1]
    return (last["max_tile"], last["score"], last["step"])


def format_bar(max_tile: int, width: int = 30) -> str:
    """Log-scale progress bar: each tile doubling advances one notch."""
    import math
    filled = min(width, int(math.log2(max(2, max_tile))) * (width // 12))
    return "#" * filled + "-" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pkl_path", nargs="?", default=None,
                        help="Champion .pkl (default: latest TwentyFortyEight_*_champion.pkl)")
    parser.add_argument("num_games", nargs="?", type=int, default=50,
                        help="How many games to play (default 50)")
    parser.add_argument("--mcts", action="store_true",
                        help="Use full u-MCTS at inference (slower, worse for "
                             "this champion since training evaluated greedy)")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Safety cap on game length")
    parser.add_argument("--fps", type=int, default=6,
                        help="Output framerate (default 6)")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="Seed of game 0; game i uses seed_base+i. "
                             "Pass a fixed value for reproducibility.")
    parser.add_argument("--out", default=None,
                        help="Output path (default: runs/<prefix>_viz/best_of_<N>.mp4)")
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
    mode_str = "u-MCTS" if args.mcts else "greedy (no MCTS)"
    print(f"Mode:      {mode_str}")
    print(f"Games:     {args.num_games}  (seed {args.seed_base}..{args.seed_base + args.num_games - 1})")
    print()

    action_space = None
    best_frames  = None
    best_key     = (-1, -1, -1)
    best_idx     = -1
    max_tiles    = []
    scores       = []

    for i in range(args.num_games):
        seed = args.seed_base + i
        frames, action_space = play_game(
            nnm, use_mcts=args.mcts,
            max_steps=args.max_steps, seed=seed,
        )
        last = frames[-1]
        max_tiles.append(last["max_tile"])
        scores.append(last["score"])

        key = score_game(frames)
        marker = " "
        if key > best_key:
            best_key    = key
            best_frames = frames
            best_idx    = i
            marker = "*"

        bar = format_bar(last["max_tile"])
        print(f"  {marker} Game {i + 1:3d}/{args.num_games}: "
              f"max_tile={last['max_tile']:5d}  score={last['score']:6d}  "
              f"steps={last['step']:4d}  [{bar}]")

    # Summary across all N games -- lets the viewer see whether the best
    # game we're about to render is an outlier or typical upper-tail.
    print()
    print(f"  Across {args.num_games} games:")
    print(f"    max tile    avg={np.mean(max_tiles):6.1f}  "
          f"median={int(np.median(max_tiles)):5d}  "
          f"max={int(np.max(max_tiles)):5d}  min={int(np.min(max_tiles)):5d}")
    print(f"    game score  avg={np.mean(scores):7.1f}  "
          f"max={int(np.max(scores)):6d}")
    print()
    print(f"  Best game: #{best_idx + 1}  "
          f"max_tile={best_key[0]}  score={best_key[1]}  steps={best_key[2]}")

    viz_dir = derive_viz_dir(pkl_path)
    os.makedirs(viz_dir, exist_ok=True)
    out_path = args.out or os.path.join(
        viz_dir, f"best_of_{args.num_games}.mp4"
    )

    print(f"\nRendering {len(best_frames)} frames at {args.fps} fps...")
    final_path = render_video(
        best_frames, action_space, out_path, fps=args.fps,
        title_suffix=(f"  -  best of {args.num_games}  ({mode_str})"),
        policy_mode=("mcts" if args.mcts else "greedy"),
    )
    print(f"\n  Saved -> {os.path.abspath(final_path)}")


if __name__ == "__main__":
    main()
