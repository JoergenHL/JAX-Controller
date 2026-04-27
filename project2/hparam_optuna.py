#!/usr/bin/env python3
"""Overnight training harness using Optuna TPE for diversified search.

Runs N_TRIALS training configs sampled by Optuna's Bayesian TPE sampler.
Each trial that survives pruning saves checkpoints + a top-K leaderboard;
the best entry from each trial's leaderboard becomes that trial's champion.
Afterwards, meta_shootout.py compares all trial champions on 1000 games
to pick the overnight winner.

WHY THIS REDUCES LOCAL-OPTIMA RISK
  - Each trial has a different random weight initialization
  - Each trial has different hparams (lr, c, d_max, dir_epsilon, eps_per_iter)
  - Meta-shootout on 1000 games per candidate discriminates robust winners
    from the lucky ones that happened to score well on a short 10-game eval

SEARCH SPACE — narrowed around Trial 24's basin (prior run's best, avg=114)
  lr:           log-uniform [5e-4, 2e-3]
  c:            log-uniform [0.3,  0.8]
  d_max:        int         [9,   13]
  eps_per_iter: int         [8,   10]
  dir_epsilon:  uniform     [0.15, 0.30]
  (fixed: num_sims=50, abstract_dim=32, width=128, n_hidden_layers=2)

Networks use 2 hidden layers per net (not 3 like config.py) to stay consistent
with the seeded PRIOR_TRIALS history below. Switching to 3 layers would make
the TPE priors misleading since they were recorded at 2 layers.

OUTPUTS (all under runs/optuna_<ts>/)
    trial_NN_ckpt_iterXXX.pkl + .json   # snapshots for progression video
    trial_NN_bestK.pkl + .json          # top-K leaderboard during training
    trial_NN_champion.pkl + .json       # trial winner (after per-trial shootout)
    trial_NN_ranking.json               # per-trial leaderboard ranking

RUN
    ./run_overnight.sh                  # creates conda env + launches this

Pruning: a trial whose last PRUNE_TAIL pre-run evals average below
PRUNE_THRESHOLD is cancelled before committing to the full budget. The tail
mean (not the whole pre-run mean) is used because the first few iters are
pure exploration noise — averaging them in would penalise trials that are
climbing on-trajectory. Pre-run uses run_prefix=None so nothing is written
to disk for a trial that turns out to be unpromising.
"""

import csv
import json
import os
import shutil
import time
from datetime import datetime

import optuna
from optuna.trial import FrozenTrial, TrialState

# ── Config ─────────────────────────────────────────────────────────────────────
GAME            = "CartPole"
UPDATES         = 50
MINIBATCH       = 64
ROLL_AHEAD      = 3
GAMMA           = 0.99
NUM_WORKERS     = 3

# Budget per trial
FULL_ITERS            = 25   # total iters if trial survives the prune gate
PRUNE_AT              = 8    # eval avg checked at this iteration (~32% of budget)
PRUNE_THRESHOLD       = 20.0 # prune if mean of last PRUNE_TAIL pre-run evals < this
PRUNE_TAIL            = 3    # only the last-N iters inform the prune decision (tail
                             # is a stronger signal than the full mean — early iters
                             # are pure exploration noise on CartPole)
CHECKPOINT_EVERY      = 5    # save checkpoint pkl every N iters (post-prune only)
LEADERBOARD_K         = 3    # top-K during training
LEADERBOARD_THRESHOLD = 25.0 # min eval avg to enter leaderboard (lowered for 25-iter
                             # budget — agents rarely cross 40 that fast)
EVAL_GAMES            = 10   # per-iter eval games (fast; noisy signal for leaderboard)
FINAL_EVAL_PER_TRIAL  = 200  # games used in the per-trial champion shootout

# Total trials. Budget notes (CartPole, 3 workers, Apr 19 overnight timings):
#   25 iters unpruned → ~60 min per trial (~2.4 min/iter)
#   pruned at iter 8  → ~20 min per trial
# At ~40% prune rate, 15 trials lands around 9–10h. Drop to 10 for a shorter
# window (~6–7h) or raise to 20 for an overnight (~12–13h).
N_TRIALS       = 15


# ── Output directory ───────────────────────────────────────────────────────────
# NB: RUN_DIR is computed at import time but the directory itself is only
# created inside main() — a module import that crashes before main() runs
# (e.g. optuna missing, config typo) shouldn't leave an empty optuna_<ts>/ dir
# cluttering runs/. meta_shootout.py globs optuna_*/ directories and would
# otherwise pick up empty ones as candidate runs.
RUN_TS     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR    = os.path.join("runs", f"optuna_{RUN_TS}")

CSV_PATH   = os.path.join(RUN_DIR, "trials.csv")
FIELDS     = [
    "trial", "lr", "c", "d_max", "num_sims", "eps_per_iter",
    "abstract_dim", "width", "dir_epsilon",
    "selection_avg", "final_avg", "final_max",
    "final_pol_loss", "final_val_loss",
    "iter_avgs", "pruned", "duration_min",
    "champion_pkl",
]


# ── Prior results (seed TPE with known observations) ──────────────────────────
# Format: (lr, c, d_max, num_sims, eps_per_iter, abstract_dim, width, dir_epsilon, agent_avg)
#
# All priors must be measured at the SAME budget TPE now sees (FULL_ITERS), or
# TPE will model a score distribution that this harness can't reproduce and
# waste its early trials chasing phantom optima. Older priors from the 100-iter
# overnight and earlier exploratory rounds have been dropped for that reason.
#
# Below: last night's 5 completed trials (optuna_2026-04-18_23-25-07), scored
# by mean of iters 23-25 of their iter_avgs — i.e. what each config would have
# reported if the run had stopped at FULL_ITERS=25. Trial 3 was pruned before
# iter 25 and is omitted.
PRIOR_TRIALS = [
    (0.0006682, 0.6596, 11, 50,  8, 32, 128, 0.165, 17.67),   # Apr19 trial 1
    (0.0007812, 0.4086, 11, 50, 10, 32, 128, 0.229, 24.37),   # Apr19 trial 2
    (0.0011802, 0.4282, 13, 50, 10, 32, 128, 0.274, 27.50),   # Apr19 trial 4 (eventual overnight winner)
    (0.0012363, 0.4211, 13, 50, 10, 32, 128, 0.281, 19.53),   # Apr19 trial 5
    (0.0013489, 0.4242, 12, 50, 10, 32, 128, 0.275, 37.63),   # Apr19 trial 6 (best at 25 iters)
]


# ── Search-space distributions ────────────────────────────────────────────────
import optuna.distributions as D
DISTRIBUTIONS = {
    "lr":           D.FloatDistribution(5e-4, 2e-3, log=True),
    "c":            D.FloatDistribution(0.3,  0.8,  log=True),
    "d_max":        D.IntDistribution(9, 13),
    "num_sims":     D.CategoricalDistribution([50]),
    "eps_per_iter": D.IntDistribution(8, 10),
    "abstract_dim": D.CategoricalDistribution([32]),
    "width":        D.CategoricalDistribution([128]),
    "dir_epsilon":  D.FloatDistribution(0.15, 0.30),
}


def seed_study(study: optuna.Study) -> None:
    """Inject every PRIOR_TRIALS row as a completed FrozenTrial.

    Clips out-of-range values to the current search-space boundaries so Optuna
    accepts them — they still inform TPE's density model.
    """
    now = datetime.now()
    seeded = 0
    for i, row in enumerate(PRIOR_TRIALS):
        lr, c, d_max, num_sims, eps, abstract_dim, width, dir_epsilon, score = row

        params = {
            "lr":           max(5e-4, min(2e-3, lr)),
            "c":            max(0.3,  min(0.8,  c)),
            "d_max":        max(9,    min(13,   d_max)),
            "num_sims":     num_sims if num_sims in [50] else 50,
            "eps_per_iter": max(8,    min(10,   eps)),
            "abstract_dim": abstract_dim if abstract_dim in [32] else 32,
            "width":        width if width in [128] else 128,
            "dir_epsilon":  max(0.15, min(0.30, dir_epsilon)),
        }

        trial = FrozenTrial(
            number=i,
            state=TrialState.COMPLETE,
            value=score,
            params=params,
            distributions=DISTRIBUTIONS,
            intermediate_values={},
            datetime_start=now,
            datetime_complete=now,
            trial_id=i,
            user_attrs={},
            system_attrs={},
        )
        study.add_trial(trial)
        seeded += 1

    best = max(PRIOR_TRIALS, key=lambda x: x[-1])
    print(f"  Seeded study with {seeded} prior trials.")
    print(f"  Best prior: lr={best[0]}, c={best[1]}, d_max={best[2]}, "
          f"sims={best[3]}, eps={best[4]}  →  avg={best[-1]}")


# ── Per-trial training ────────────────────────────────────────────────────────

def apply_trial_config(lr, c, d_max, num_sims, eps_per_iter,
                       abstract_dim, width, dir_epsilon, num_iterations):
    """Overwrite the config module in-place for this trial.

    Sequential-only — each trial runs in the same process.
    """
    import config

    config.game["name"]                  = GAME
    config.mcts["c"]                     = c
    config.mcts["num_simulations"]       = num_sims
    config.mcts["d_max"]                 = d_max
    config.mcts["dir_epsilon"]           = dir_epsilon
    config.nn["abstract_dim"]            = abstract_dim
    config.nn["nnr_hidden"]              = [width, width]
    config.nn["nnp_hidden"]              = [width, width]
    config.nn["nnd_hidden"]              = [width, width]
    config.nn["learning_rate"]           = lr
    config.training["num_iterations"]    = num_iterations
    config.training["episodes_per_iter"] = eps_per_iter
    config.training["updates_per_iter"]  = UPDATES
    config.training["minibatch_size"]    = MINIBATCH
    config.training["buffer_size"]       = eps_per_iter * 25
    config.training["roll_ahead"]        = ROLL_AHEAD
    config.training["gamma"]             = GAMMA
    config.training["num_workers"]       = NUM_WORKERS
    # Eval needed both for leaderboard candidacy and for the prune-gate check
    config.viz["eval_every"]             = 1
    config.viz["eval_games"]             = EVAL_GAMES
    config.viz["compare_baseline"]       = False
    config.viz["show_policy_analysis"]   = False
    config.viz["replay_after_training"]  = False


def build_rlm():
    """Create a fresh game + NNManager + rlm instance under the current config."""
    import config
    from nn.NNManager import NNManager
    from rlm import ReinforcementLearningManager
    from worker import _get_game

    game = _get_game(GAME)
    s, a, d = game.state_dim, game.num_actions, config.nn["abstract_dim"]

    q        = config.nn.get("q", 0)
    nnr_dims = [s * (q + 1)] + config.nn["nnr_hidden"] + [d]
    nnp_dims = [d]     + config.nn["nnp_hidden"] + [1 + a]
    nnd_dims = [d + a] + config.nn["nnd_hidden"] + [d + 1]

    nnm = NNManager()
    nnm.create_net("nnr", nnr_dims)
    nnm.create_net("nnd", nnd_dims)
    nnm.create_net("nnp", nnp_dims)

    rlm = ReinforcementLearningManager(game, nnm)
    network_dims = {"nnr": nnr_dims, "nnp": nnp_dims, "nnd": nnd_dims}
    return game, nnm, rlm, network_dims


def rename_phase2_checkpoints(trial_prefix: str, offset: int,
                               ckpt_every: int, phase2_iters: int):
    """Rename phase-2 checkpoint files so their iter number reflects the true
    iteration across the whole trial, not phase-2's local counter.

    Phase 2 wrote `_ckpt_iter{k}.pkl` for k = 5, 10, ..., phase2_iters.
    True iter = PRUNE_AT + k. This also updates the `iteration` field inside
    the companion JSON files so metadata stays consistent.
    """
    for k in range(ckpt_every, phase2_iters + 1, ckpt_every):
        true_iter = offset + k
        old_pkl  = f"{trial_prefix}_ckpt_iter{k:03d}.pkl"
        new_pkl  = f"{trial_prefix}_ckpt_iter{true_iter:03d}.pkl"
        old_json = f"{trial_prefix}_ckpt_iter{k:03d}.json"
        new_json = f"{trial_prefix}_ckpt_iter{true_iter:03d}.json"

        if os.path.exists(old_pkl):
            os.rename(old_pkl, new_pkl)
        if os.path.exists(old_json):
            with open(old_json) as f:
                meta = json.load(f)
            meta["iteration"] = true_iter
            with open(new_json, "w") as f:
                json.dump(meta, f, indent=2)
            if new_json != old_json and os.path.exists(old_json):
                os.remove(old_json)


def run_trial(trial_num: int, params: dict,
              csv_writer, csv_file) -> dict:
    """Execute one Optuna trial: pre-run → prune? → full-run → mini-shootout.

    Returns the CSV row plus the float score that Optuna's `tell()` will
    receive. The score is the champion's final_avg (a 200-game eval) if the
    trial completed, else the pre-run's average (noisy but not misleading —
    it's what got the trial pruned).
    """
    lr           = params["lr"]
    c            = params["c"]
    d_max        = params["d_max"]
    num_sims     = params["num_sims"]
    eps_per_iter = params["eps_per_iter"]
    abstract_dim = params["abstract_dim"]
    width        = params["width"]
    dir_epsilon  = params["dir_epsilon"]

    trial_prefix = os.path.join(RUN_DIR, f"trial_{trial_num:02d}")

    # ── Pre-run (pruning gate) — no disk writes ──────────────────────────────
    apply_trial_config(lr, c, d_max, num_sims, eps_per_iter,
                       abstract_dim, width, dir_epsilon,
                       num_iterations=PRUNE_AT)
    _, nnm, rlm, network_dims = build_rlm()

    # Span the cosine LR schedule over the *full* budget (pre-run + phase 2),
    # not just the pre-run's 10 iters. Without this the LR would decay to the
    # floor during pre-run and phase 2 would inherit a dead optimizer.
    full_total_updates = FULL_ITERS * UPDATES

    t0 = time.time()
    result_pre = rlm.train(total_updates_override=full_total_updates)   # run_prefix=None → nothing hits disk
    pre_evals  = result_pre.get("eval_scores", [])
    pre_avg    = (sum(e[2] for e in pre_evals) / len(pre_evals)) if pre_evals else 0.0
    # Prune decision uses the *tail* of the pre-run: the last PRUNE_TAIL evals.
    # Averaging across all 10 iters drags the signal down with early exploration
    # noise — a trial that climbs from 10 → 40 reads as avg 25 and gets cut
    # despite being on-trajectory. The tail sees where the trial is heading.
    tail         = pre_evals[-PRUNE_TAIL:] if pre_evals else []
    prune_signal = (sum(e[2] for e in tail) / len(tail)) if tail else 0.0

    pruned = prune_signal < PRUNE_THRESHOLD

    if pruned:
        result       = result_pre
        selection_avg = pre_avg
        final_avg, final_scores = pre_avg, []
        champion_pkl = ""
    else:
        # ── Phase 2 — full-budget training with checkpoints + leaderboard ────
        # Preserve weights, opt_state and replay buffer from the pre-run by
        # reusing the same rlm instance; just reconfigure and call train again.
        import config
        config.training["num_iterations"]     = FULL_ITERS - PRUNE_AT
        config.viz["checkpoint_every"]        = CHECKPOINT_EVERY
        config.viz["best_leaderboard_k"]      = LEADERBOARD_K
        config.viz["best_threshold"]          = LEADERBOARD_THRESHOLD
        # final_eval_games is read inside train_system.py, not rlm.train;
        # per-trial shootout below uses FINAL_EVAL_PER_TRIAL directly.
        config.viz["final_eval_games"]        = FINAL_EVAL_PER_TRIAL

        result_post = rlm.train(
            run_prefix             = trial_prefix,
            network_dims           = network_dims,
            game_name              = GAME,
            total_updates_override = full_total_updates,  # same schedule span as pre-run
        )

        result = {
            "losses":      result_pre["losses"]      + result_post["losses"],
            "eval_scores": result_pre.get("eval_scores", []) +
                            result_post.get("eval_scores", []),
        }

        # Rename phase-2 checkpoints so iter numbers are consistent with the
        # full-run timeline (see rename_phase2_checkpoints docstring).
        rename_phase2_checkpoints(
            trial_prefix,
            offset=PRUNE_AT,
            ckpt_every=CHECKPOINT_EVERY,
            phase2_iters=FULL_ITERS - PRUNE_AT,
        )

        # ── Per-trial mini-shootout: pick champion from leaderboard ─────────
        # Offset leaderboard iters by PRUNE_AT: rlm.train's counter resets each
        # call, so phase-2 entries carry local iter numbers (1..phase2_iters).
        # The user-facing ranking should show true iters along the full run.
        leaderboard = result_post.get("leaderboard", [])
        if leaderboard:
            ranking = []
            for entry in leaderboard:
                nnm.set_layer_weights(entry["layer_weights"])
                _, fa, fs = rlm.evaluate(num_games=FINAL_EVAL_PER_TRIAL)
                ranking.append({
                    "iteration":          entry["iteration"] + PRUNE_AT,
                    "selection_eval_avg": round(float(entry["eval_avg"]), 2),
                    "final_eval_avg":     round(float(fa), 2),
                    "final_eval_games":   FINAL_EVAL_PER_TRIAL,
                    "pkl_path":           entry.get("pkl_path"),
                })
            ranking.sort(key=lambda e: e["final_eval_avg"], reverse=True)
            for rank, e in enumerate(ranking, start=1):
                e["rank"] = rank

            # Persist ranking + copy top entry to <trial>_champion.pkl/.json
            ranking_path = f"{trial_prefix}_ranking.json"
            with open(ranking_path, "w") as f:
                json.dump({
                    "trial": trial_num,
                    "params": params,
                    "final_eval_games": FINAL_EVAL_PER_TRIAL,
                    "ranking": ranking,
                }, f, indent=2)

            winner = ranking[0]
            champion_pkl  = f"{trial_prefix}_champion.pkl"
            champion_json = f"{trial_prefix}_champion.json"
            if winner["pkl_path"] and os.path.exists(winner["pkl_path"]):
                shutil.copy2(winner["pkl_path"], champion_pkl)
                src_json = winner["pkl_path"].replace(".pkl", ".json")
                if os.path.exists(src_json):
                    with open(src_json) as f:
                        champ_meta = json.load(f)
                    champ_meta["kind"] = "champion"
                    champ_meta["iteration"]        = winner["iteration"]
                    champ_meta["trial"]            = trial_num
                    champ_meta["final_eval_avg"]   = winner["final_eval_avg"]
                    champ_meta["final_eval_games"] = FINAL_EVAL_PER_TRIAL
                    with open(champion_json, "w") as f:
                        json.dump(champ_meta, f, indent=2)

            final_avg    = winner["final_eval_avg"]
            final_scores = []
            selection_avg = winner["selection_eval_avg"]
        else:
            # No leaderboard entry cleared the threshold — fall back to
            # evaluating the final-iter weights directly.
            _, final_avg, final_scores = rlm.evaluate(num_games=FINAL_EVAL_PER_TRIAL)
            selection_avg = pre_avg
            champion_pkl = ""

    duration_min = (time.time() - t0) / 60
    losses       = result["losses"]
    _, fv, fp, _ = losses[-1] if losses else (0, 0, 0, 0)
    iter_avgs    = [round(e[2], 1) for e in result.get("eval_scores", [])]

    row = {
        "trial":          trial_num,
        "lr":             lr,
        "c":              c,
        "d_max":          d_max,
        "num_sims":       num_sims,
        "eps_per_iter":   eps_per_iter,
        "abstract_dim":   abstract_dim,
        "width":          width,
        "dir_epsilon":    round(dir_epsilon, 3),
        "selection_avg":  round(selection_avg, 2),
        "final_avg":      round(final_avg, 2),
        "final_max":      (max(final_scores) if final_scores else 0),
        "final_pol_loss": round(fp, 4),
        "final_val_loss": round(fv, 4),
        "iter_avgs":      str(iter_avgs),
        "pruned":         pruned,
        "duration_min":   round(duration_min, 1),
        "champion_pkl":   champion_pkl,
    }
    csv_writer.writerow(row)
    csv_file.flush()
    return row, final_avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    print("=" * 76)
    print("OVERNIGHT TRAINING — Optuna TPE harness")
    print(f"  Run dir: {RUN_DIR}")
    print(f"  Trials:  {N_TRIALS}  |  Budget: {FULL_ITERS} iters each"
          f"  |  Prune if mean of last {PRUNE_TAIL} pre-run evals < {PRUNE_THRESHOLD} "
          f"(check at iter {PRUNE_AT})")
    print(f"  Seeding: {len(PRIOR_TRIALS)} prior trials")
    print("  Search space:")
    for name, dist in DISTRIBUTIONS.items():
        print(f"    {name:<14} {dist}")
    print("=" * 76)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=3,   # few random trials (we're already seeded)
            seed=42,
        ),
    )
    seed_study(study)

    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        csv_file.flush()

        for i in range(1, N_TRIALS + 1):
            opt_trial = study.ask()
            params = {
                "lr":           opt_trial.suggest_float("lr",  5e-4, 2e-3, log=True),
                "c":            opt_trial.suggest_float("c",   0.3,  0.8,  log=True),
                "d_max":        opt_trial.suggest_int("d_max", 9, 13),
                "num_sims":     opt_trial.suggest_categorical("num_sims",     [50]),
                "eps_per_iter": opt_trial.suggest_int("eps_per_iter", 8, 10),
                "abstract_dim": opt_trial.suggest_categorical("abstract_dim", [32]),
                "width":        opt_trial.suggest_categorical("width",        [128]),
                "dir_epsilon":  opt_trial.suggest_float("dir_epsilon", 0.15, 0.30),
            }

            print(f"\n[{i:2d}/{N_TRIALS}]  "
                  f"lr={params['lr']:.5f}  c={params['c']:.2f}  "
                  f"d_max={params['d_max']}  sims={params['num_sims']}  "
                  f"eps={params['eps_per_iter']}  ε={params['dir_epsilon']:.2f}")

            try:
                row, score = run_trial(i, params, writer, csv_file)
                study.tell(opt_trial, score)

                status = "PRUNED" if row["pruned"] else "full"
                print(f"  [{status}] selection={row['selection_avg']:6.2f}  "
                      f"final={row['final_avg']:6.2f}  "
                      f"{row['duration_min']:.1f} min")
                if row["champion_pkl"]:
                    print(f"  → {row['champion_pkl']}")

                best = study.best_trial
                print(f"  Best in study (incl. seeded): avg={best.value:.2f}")

            except Exception as e:
                study.tell(opt_trial, state=TrialState.FAIL)
                print(f"  ✗ Failed: {e}")
                import traceback; traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print(f"DONE  — results in {CSV_PATH}")
    print(f"  Champions saved under: {RUN_DIR}/trial_*_champion.pkl")
    print(f"  Next step: conda run -n AI python meta_shootout.py {RUN_DIR}")
    print("=" * 76)


if __name__ == "__main__":
    main()
