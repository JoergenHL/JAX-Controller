"""Experiment logger: saves hyperparameters + per-iteration metrics to JSON.

Each call to train_system.py produces one JSON file under runs/ named:
    <GameClass>_<YYYY-MM-DD_HH-MM-SS>.json

The file records the full config snapshot, network shapes, and every
evaluation checkpoint so runs can be compared without re-running.
"""

import json
import os
from datetime import datetime


class RunLogger:
    """Accumulate metrics for one training run and persist to JSON."""

    def __init__(self, game, config_module, network_dims: dict,
                 timestamp_str: str = None):
        """
        Args:
            game:           game instance (for class name + reward_scale)
            config_module:  the imported config module (for snapshot)
            network_dims:   dict with 'nnr', 'nnp', 'nnd' dimension lists
            timestamp_str:  optional pre-derived timestamp string
                            ("YYYY-MM-DD_HH-MM-SS"). When supplied, the
                            logger, interval checkpoints, and best-leaderboard
                            files all share the same run prefix.
        """
        ts = timestamp_str or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_name = f"{game.__class__.__name__}_{ts}"

        self.data = {
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "game":         game.__class__.__name__,
            "reward_scale": game.reward_scale,
            "config": {
                "mcts":     dict(config_module.mcts),
                "nn":       dict(config_module.nn),
                "training": dict(config_module.training),
                "viz":      dict(config_module.viz),
            },
            "network_dims": network_dims,
            "iterations":   [],   # filled by log_run()
        }

    def log_run(self, train_result: dict):
        """Extract per-iteration records from the dict returned by rlm.train().

        Expects keys:
            losses:          list of (total, value, policy, reward) per epoch
            iter_boundaries: list of epoch indices where each iteration starts
            eval_scores:     list of (iteration, pct, avg_tile, [tiles]) — may be empty
        """
        losses          = train_result["losses"]
        boundaries      = train_result["iter_boundaries"]
        eval_scores     = train_result.get("eval_scores", [])
        eval_by_iter    = {it: (pct, avg, tiles) for it, pct, avg, tiles in eval_scores}

        num_iters = len(boundaries)
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < num_iters else len(losses)
            iter_losses = losses[start:end]

            if iter_losses:
                last = iter_losses[-1]
                final_loss = {
                    "total":  round(last[0], 4),
                    "value":  round(last[1], 4),
                    "policy": round(last[2], 4),
                    "reward": round(last[3], 4),
                }
            else:
                final_loss = None

            iteration_num = i + 1
            record = {
                "iteration":        iteration_num,
                "epochs_trained":   len(iter_losses),
                "final_epoch_loss": final_loss,
            }

            if iteration_num in eval_by_iter:
                pct, avg, scores = eval_by_iter[iteration_num]
                record["eval_pct"]    = round(pct, 1)
                record["avg_score"]   = round(avg, 1)
                record["max_score"]   = max(scores)
                record["all_scores"]  = scores

            self.data["iterations"].append(record)

    def log_eval(self, pct: float, avg: float, scores: list):
        """Attach the final post-training agent evaluation to the run record."""
        self.data["agent_eval"] = {
            "num_games":  len(scores),
            "avg_score":  round(avg, 1),
            "max_score":  max(scores),
            "win_pct":    round(pct, 1),
            "all_scores": scores,
        }

    def log_baseline(self, pct: float, avg: float, scores: list):
        """Attach one-time random-baseline results to the run record."""
        self.data["baseline"] = {
            "num_games":  len(scores),
            "avg_score":  round(avg, 1),
            "max_score":  max(scores),
            "win_pct":    round(pct, 1),
            "all_scores": scores,
        }

    def log_leaderboard(self, entries: list):
        """Attach final top-K leaderboard ranking (post-training shootout).

        Args:
            entries: list of dicts sorted best-first. Each entry:
              {
                "rank":                int,
                "iteration":           int,
                "selection_eval_avg":  float,  # 50-game avg that earned its slot
                "final_eval_avg":      float,  # 1000-game shootout avg
                "final_eval_games":    int,
                "pkl_path":            str,
              }
        """
        self.data["leaderboard"] = entries

    def log_mcts_eval(self, pct: float, avg: float, scores: list):
        """Attach one-time MCTS evaluation results to the run record."""
        self.data["mcts_eval"] = {
            "num_games":  len(scores),
            "avg_score":  round(avg, 1),
            "max_score":  max(scores),
            "win_pct":    round(pct, 1),
            "all_scores": scores,
        }

    def save(self, runs_dir: str = "runs") -> str:
        """Write JSON to runs/<run_name>.json. Returns the file path."""
        os.makedirs(runs_dir, exist_ok=True)
        path = os.path.join(runs_dir, f"{self.run_name}.json")
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"  Run saved → {path}")
        return path
