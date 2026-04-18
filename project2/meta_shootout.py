#!/usr/bin/env python3
"""Compare all trial champions from an Optuna overnight run and pick a winner.

Each Optuna trial saved a `trial_NN_champion.pkl` — the best leaderboard
entry from that trial after a per-trial 200-game shootout. This script runs
a final 1000-game greedy eval on every champion and crowns the overall winner.

WHY A SECOND SHOOTOUT
  The per-trial shootout used 200 games, which is enough to rank entries
  within a single trial but not to compare across trials: a 200-game avg has
  roughly ±3–5 standard error at CartPole scales. Cross-trial decisions
  deserve 1000 games, which cuts that to ~±1–2.

OUTPUTS
  runs/optuna_<ts>/meta_ranking.json    — full 1000-game scores per champion
  runs/overnight_champion.pkl           — copy of the winning champion
  runs/overnight_champion.json          — companion metadata (kind=overnight_champion)

USAGE
    conda run -n AI python meta_shootout.py                         # latest optuna_* dir
    conda run -n AI python meta_shootout.py runs/optuna_<ts>/       # specific dir
    conda run -n AI python meta_shootout.py runs/optuna_<ts>/ 500   # custom game count

After the overall winner is chosen, the losing trials' intermediate
checkpoint pkls are deleted to keep runs/ tidy — the winner's checkpoints
are kept so they can feed viz_progression.py (training progression video).
"""

import glob
import json
import os
import shutil
import sys
import time

FINAL_GAMES_DEFAULT = 1000


def find_latest_optuna_dir(runs_dir: str = "runs") -> str:
    candidates = sorted(glob.glob(os.path.join(runs_dir, "optuna_*")),
                        key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(
            f"No optuna_* directories in {runs_dir}/. "
            "Run hparam_optuna.py first."
        )
    return candidates[-1]


def list_champions(run_dir: str) -> list[str]:
    """Return sorted list of trial_NN_champion.pkl paths in run_dir."""
    return sorted(glob.glob(os.path.join(run_dir, "trial_*_champion.pkl")))


def load_champion_config(pkl_path: str) -> dict:
    """Read the companion JSON and return the network-shape fields we need."""
    json_path = pkl_path.replace(".pkl", ".json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No companion JSON for: {pkl_path}")
    with open(json_path) as f:
        return json.load(f)


def apply_network_shape(meta: dict) -> None:
    """Overwrite global config to match this champion's architecture.

    NNManager.load needs the right dims *before* load() fills the weights,
    so we read them from the companion JSON and set them on the config
    module, then build + load nets against that shape.
    """
    import config
    nn_cfg = meta.get("config", {}).get("nn", {})
    training_cfg = meta.get("config", {}).get("training", {})
    mcts_cfg = meta.get("config", {}).get("mcts", {})

    # Only override architecture-affecting keys. Others (eval games, leaderboard)
    # are per-shootout settings set below in evaluate_champion.
    for k in ("abstract_dim", "q", "nnr_hidden", "nnp_hidden", "nnd_hidden"):
        if k in nn_cfg:
            config.nn[k] = nn_cfg[k]
    if "num_workers" in training_cfg:
        config.training["num_workers"] = training_cfg["num_workers"]
    # MCTS hparams don't affect greedy eval, but mirror them for consistency.
    for k in ("c", "d_max", "num_simulations", "dir_alpha", "dir_epsilon"):
        if k in mcts_cfg:
            config.mcts[k] = mcts_cfg[k]


def evaluate_champion(pkl_path: str, num_games: int) -> dict:
    """Run a num_games greedy eval on one champion and return summary stats."""
    # Import lazily so every champion evaluation resets JAX state cleanly.
    import config
    from nn.NNManager import NNManager
    from rlm import ReinforcementLearningManager
    from worker import _get_game

    meta = load_champion_config(pkl_path)
    apply_network_shape(meta)

    game = _get_game(meta.get("game", "CartPole"))
    nnm  = NNManager()
    nnm.load(pkl_path)
    rlm  = ReinforcementLearningManager(game, nnm)

    t0 = time.time()
    pct, avg, scores = rlm.evaluate(num_games=num_games)
    duration_min = (time.time() - t0) / 60

    return {
        "pkl_path":    pkl_path,
        "trial":       meta.get("trial") or _trial_num_from_path(pkl_path),
        "iteration":   meta.get("iteration"),
        "eval_avg":    round(float(avg), 2),
        "eval_min":    float(min(scores)),
        "eval_max":    float(max(scores)),
        "num_games":   num_games,
        "duration_min": round(duration_min, 1),
    }


def _trial_num_from_path(pkl_path: str) -> int:
    """Extract NN from a filename of the form `.../trial_NN_champion.pkl`."""
    name = os.path.basename(pkl_path)
    # name == "trial_03_champion.pkl"
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def cleanup_losers(run_dir: str, winner_pkl: str) -> None:
    """Delete intermediate _ckpt_iter*.pkl files from losing trials.

    Keep the winner's checkpoints (needed by viz_progression.py) plus every
    trial's final champion/best/ranking files.
    """
    winner_trial = _trial_num_from_path(winner_pkl)
    for ckpt in glob.glob(os.path.join(run_dir, "trial_*_ckpt_iter*.pkl")):
        if _trial_num_from_path(ckpt) == winner_trial:
            continue
        for ext in (".pkl", ".json"):
            p = ckpt.replace(".pkl", ext)
            if os.path.exists(p):
                os.remove(p)


def main():
    run_dir   = sys.argv[1] if len(sys.argv) > 1 else find_latest_optuna_dir()
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else FINAL_GAMES_DEFAULT

    if not os.path.isdir(run_dir):
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    champions = list_champions(run_dir)
    if not champions:
        print(f"Error: no trial_*_champion.pkl files in {run_dir}")
        sys.exit(1)

    print("=" * 72)
    print(f"META-SHOOTOUT  —  {len(champions)} champions × {num_games} games")
    print(f"  Source: {run_dir}")
    print("=" * 72)

    results = []
    for i, pkl in enumerate(champions, start=1):
        print(f"\n[{i}/{len(champions)}] {os.path.basename(pkl)}")
        try:
            result = evaluate_champion(pkl, num_games)
            results.append(result)
            print(f"  avg={result['eval_avg']:.1f}  "
                  f"min={result['eval_min']:.0f}  max={result['eval_max']:.0f}  "
                  f"({result['duration_min']:.1f} min)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print("\nNo champions evaluated successfully — nothing to save.")
        sys.exit(1)

    # ── Ranking ───────────────────────────────────────────────────────────────
    results.sort(key=lambda r: r["eval_avg"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    print("\n" + "=" * 72)
    print("Final ranking:")
    print(f"  {'rank':>4}  {'trial':>5}  {'iter':>4}  {'avg':>8}  {'min':>4}  {'max':>4}")
    for r in results:
        print(f"  {r['rank']:>4}  {r['trial']:>5}  {r['iteration']:>4}  "
              f"{r['eval_avg']:>8.2f}  {r['eval_min']:>4.0f}  {r['eval_max']:>4.0f}")
    print("=" * 72)

    # Save ranking JSON
    ranking_path = os.path.join(run_dir, "meta_ranking.json")
    with open(ranking_path, "w") as f:
        json.dump({
            "run_dir":   run_dir,
            "num_games": num_games,
            "ranking":   results,
        }, f, indent=2)
    print(f"\n  Ranking → {ranking_path}")

    # Copy winner to runs/overnight_champion.{pkl,json}
    winner = results[0]
    overnight_pkl  = os.path.join("runs", "overnight_champion.pkl")
    overnight_json = os.path.join("runs", "overnight_champion.json")
    shutil.copy2(winner["pkl_path"], overnight_pkl)

    src_json = winner["pkl_path"].replace(".pkl", ".json")
    if os.path.exists(src_json):
        with open(src_json) as f:
            meta = json.load(f)
        meta["kind"] = "overnight_champion"
        meta["meta_eval_avg"]   = winner["eval_avg"]
        meta["meta_eval_games"] = num_games
        meta["source_pkl"]      = winner["pkl_path"]
        with open(overnight_json, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"  Winner → {overnight_pkl}")
    print(f"          (avg={winner['eval_avg']:.1f} over {num_games} games)")

    # Cleanup — delete losing trials' intermediate checkpoints to save disk
    cleanup_losers(run_dir, winner["pkl_path"])
    print(f"\n  Cleaned up intermediate checkpoints from losing trials.")
    print(f"  Winner's _ckpt_iter*.pkl files kept for progression video.")


if __name__ == "__main__":
    main()
