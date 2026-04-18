#!/usr/bin/env python3
"""Bayesian hyperparameter search using Optuna (TPE sampler).

Seeded with all 21 prior trials so the surrogate model starts informed.
Early stopping prunes trials that show no promise by iter PRUNE_AT.

Search space:
    lr            log-uniform [5e-4, 3e-3]   (narrowed from evidence)
    c             log-uniform [0.3,  3.0]     (c>4 showed no benefit)
    d_max         int         [8,   15]       (below 5 bad, diminishing above 15)
    num_sims      categorical [30, 50, 75]
    eps_per_iter  int         [6,  10]        (18 confirmed bad)
    abstract_dim  categorical [16, 32, 64]    (latent space capacity — never tuned)
    width         categorical [64, 128]       (hidden layer width for all 3 nets)
    dir_epsilon   uniform     [0.05, 0.40]    (noise fraction at root — may drive
                                               uniform policy loss)

Fixed (confirmed best or infrastructure):
    updates=50, minibatch=64, roll_ahead=3, gamma=0.99, workers=3

Prior trials used abstract_dim=32, width=128, dir_epsilon=0.25 (config defaults).
These are recorded as such in the seeded history so TPE treats them as observations
at those values, not unknowns.

Usage:
    conda run -n AI python hparam_optuna.py
"""

import csv
import os
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
EVAL_GAMES      = 20

N_TRIALS        = 30       # new trials to run (on top of seeded history)
FULL_ITERS      = 25       # budget per trial if not pruned
PRUNE_AT        = 8        # iter at which we check for early stopping
PRUNE_THRESHOLD = 25.0     # prune if avg across first PRUNE_AT iters < this

# ── Output ─────────────────────────────────────────────────────────────────────
os.makedirs("runs", exist_ok=True)
ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_path = f"runs/hparam_optuna_{ts}.csv"
FIELDS   = [
    "trial", "lr", "c", "d_max", "num_sims", "eps_per_iter",
    "abstract_dim", "width", "dir_epsilon",
    "agent_avg", "agent_max", "final_pol_loss", "final_val_loss",
    "iter_avgs", "pruned", "duration_min",
]


# ── Prior results ───────────────────────────────────────────────────────────────
# All prior trials. New params (abstract_dim, width, dir_epsilon) are filled with
# the config.py defaults that were in use at the time: 32, 128, 0.25.
#
# Format: (lr, c, d_max, num_sims, eps_per_iter, abstract_dim, width, dir_epsilon, agent_avg)
PRIOR_TRIALS = [
    # ── Round 1: lr × d_max × sims (10 iters, 9 eps, c=2.0) ──────────────────
    (0.001,  2.0,  5, 30,  9, 32, 128, 0.25, 16.95),
    (0.001,  2.0,  5, 50,  9, 32, 128, 0.25, 15.35),
    (0.001,  2.0, 10, 30,  9, 32, 128, 0.25, 22.90),
    (0.001,  2.0, 10, 50,  9, 32, 128, 0.25, 16.00),
    (0.0003, 2.0,  5, 30,  9, 32, 128, 0.25, 16.95),
    (0.0003, 2.0,  5, 50,  9, 32, 128, 0.25,  9.80),
    (0.0003, 2.0, 10, 30,  9, 32, 128, 0.25, 10.75),
    (0.0003, 2.0, 10, 50,  9, 32, 128, 0.25, 24.40),
    (0.0001, 2.0,  5, 30,  9, 32, 128, 0.25,  9.35),
    (0.0001, 2.0,  5, 50,  9, 32, 128, 0.25,  9.35),
    (0.0001, 2.0, 10, 30,  9, 32, 128, 0.25, 11.10),
    (0.0001, 2.0, 10, 50,  9, 32, 128, 0.25, 25.80),
    # ── Round 2: lr sweep + eps (25 iters phase-1, 20 iters phase-2, c=2.0) ──
    (0.001,  2.0, 10, 50,  9, 32, 128, 0.25, 59.25),
    (0.0003, 2.0, 10, 50,  9, 32, 128, 0.25, 46.25),
    (0.0001, 2.0, 10, 50,  9, 32, 128, 0.25,  9.45),
    (0.001,  2.0, 10, 50, 18, 32, 128, 0.25, 15.45),
    (0.0001, 2.0, 10, 50, 18, 32, 128, 0.25,  9.35),
    # ── c search: lr=0.001, d_max=10, sims=50, 9 eps, 10 iters ───────────────
    (0.001,  0.5, 10, 50,  9, 32, 128, 0.25, 31.75),
    (0.001,  1.0, 10, 50,  9, 32, 128, 0.25, 18.60),
    (0.001,  2.0, 10, 50,  9, 32, 128, 0.25, 17.15),
    (0.001,  4.0, 10, 50,  9, 32, 128, 0.25, 35.20),
    (0.001,  8.0, 10, 50,  9, 32, 128, 0.25, 19.85),
]


def seed_study(study: optuna.Study) -> None:
    """Add all prior results to the study as completed FrozenTrials."""
    import optuna.distributions as D

    distributions = {
        "lr":           D.FloatDistribution(5e-4, 3e-3, log=True),
        "c":            D.FloatDistribution(0.3,  3.0,  log=True),
        "d_max":        D.IntDistribution(8, 15),
        "num_sims":     D.CategoricalDistribution([30, 50, 75]),
        "eps_per_iter": D.IntDistribution(6, 10),
        "abstract_dim": D.CategoricalDistribution([16, 32, 64]),
        "width":        D.CategoricalDistribution([64, 128]),
        "dir_epsilon":  D.FloatDistribution(0.05, 0.40),
    }

    now = datetime.now()
    seeded = 0
    for i, row in enumerate(PRIOR_TRIALS):
        lr, c, d_max, num_sims, eps, abstract_dim, width, dir_epsilon, score = row

        # Clip params that fall outside the (narrowed) search space so Optuna
        # accepts them — they still inform the model, just at boundary values.
        params = {
            "lr":           max(5e-4, min(3e-3, lr)),
            "c":            max(0.3,  min(3.0,  c)),
            "d_max":        max(8,    min(15,   d_max)),
            "num_sims":     num_sims if num_sims in [30, 50, 75] else 50,
            "eps_per_iter": max(6,    min(10,   eps)),
            "abstract_dim": abstract_dim,
            "width":        width,
            "dir_epsilon":  dir_epsilon,
        }

        trial = FrozenTrial(
            number=i,
            state=TrialState.COMPLETE,
            value=score,
            params=params,
            distributions=distributions,
            intermediate_values={},
            datetime_start=now,
            datetime_complete=now,
            trial_id=i,
            user_attrs={},
            system_attrs={},
        )
        study.add_trial(trial)
        seeded += 1

    print(f"  Seeded study with {seeded} prior trials.")
    best = max(PRIOR_TRIALS, key=lambda x: x[-1])
    print(f"  Best prior: lr={best[0]}, c={best[1]}, d_max={best[2]}, "
          f"sims={best[3]}, eps={best[4]}  →  avg={best[-1]}")


def run_trial(trial_num, lr, c, d_max, num_sims, eps_per_iter,
              abstract_dim, width, dir_epsilon, csv_writer, csv_file):
    import config
    from nn.NNManager import NNManager
    from rlm import ReinforcementLearningManager
    from worker import _get_game

    buffer_size = eps_per_iter * 25   # ~25 iters of history

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
    config.training["num_iterations"]    = FULL_ITERS
    config.training["episodes_per_iter"] = eps_per_iter
    config.training["updates_per_iter"]  = UPDATES
    config.training["minibatch_size"]    = MINIBATCH
    config.training["buffer_size"]       = buffer_size
    config.training["roll_ahead"]        = ROLL_AHEAD
    config.training["gamma"]             = GAMMA
    config.training["num_workers"]       = NUM_WORKERS
    config.viz["eval_every"]             = 1
    config.viz["eval_games"]             = 10
    config.viz["compare_baseline"]       = False
    config.viz["show_policy_analysis"]   = False
    config.viz["replay_after_training"]  = False

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

    t0 = time.time()

    # ── Phase 1: run PRUNE_AT iters and check for early stopping ──────────────
    config.training["num_iterations"] = PRUNE_AT
    result_early = rlm.train()

    eval_scores_early = result_early.get("eval_scores", [])
    early_avg = (sum(e[2] for e in eval_scores_early) / len(eval_scores_early)
                 if eval_scores_early else 0.0)

    pruned = False
    if early_avg < PRUNE_THRESHOLD:
        pruned = True
        result = result_early
    else:
        # ── Phase 2: continue with remaining iters (weights intact) ───────────
        config.training["num_iterations"] = FULL_ITERS - PRUNE_AT
        result_rest = rlm.train()
        result = {
            "losses":      result_early["losses"] + result_rest["losses"],
            "eval_scores": result_early.get("eval_scores", []) +
                           result_rest.get("eval_scores", []),
        }

    _, agent_avg, agent_scores = rlm.evaluate(num_games=EVAL_GAMES)
    duration_min = (time.time() - t0) / 60

    losses = result["losses"]
    _, final_val, final_pol, _ = losses[-1] if losses else (0, 0, 0, 0)
    iter_avgs = [round(e[2], 1) for e in result.get("eval_scores", [])]

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
        "agent_avg":      round(agent_avg, 2),
        "agent_max":      max(agent_scores),
        "final_pol_loss": round(final_pol, 4),
        "final_val_loss": round(final_val, 4),
        "iter_avgs":      str(iter_avgs),
        "pruned":         pruned,
        "duration_min":   round(duration_min, 1),
    }
    csv_writer.writerow(row)
    csv_file.flush()
    return row, agent_avg


def main():
    print("=" * 76)
    print("BAYESIAN HYPERPARAMETER SEARCH — Optuna TPE")
    print("  Search space:")
    print("    lr          log-uniform [5e-4, 3e-3]")
    print("    c           log-uniform [0.3,  3.0]")
    print("    d_max       int         [8,   15]")
    print("    num_sims    categorical [30, 50, 75]")
    print("    eps_per_iter int        [6,  10]")
    print("    abstract_dim categorical [16, 32, 64]   ← new")
    print("    width        categorical [64, 128]       ← new")
    print("    dir_epsilon  uniform    [0.05, 0.40]     ← new")
    print(f"  Seeding: {len(PRIOR_TRIALS)} prior trials")
    print(f"  New trials: {N_TRIALS}  |  Budget: {FULL_ITERS} iters  "
          f"|  Prune at iter {PRUNE_AT} if avg < {PRUNE_THRESHOLD}")
    print(f"  Results → {csv_path}")
    print("=" * 76)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=5,   # random before TPE model kicks in
            seed=42,
        ),
    )

    seed_study(study)

    results = []
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        csv_file.flush()

        for i in range(1, N_TRIALS + 1):
            opt_trial    = study.ask()
            lr           = opt_trial.suggest_float("lr",    5e-4, 3e-3, log=True)
            c            = opt_trial.suggest_float("c",     0.3,  3.0,  log=True)
            d_max        = opt_trial.suggest_int("d_max",   8,    15)
            num_sims     = opt_trial.suggest_categorical("num_sims",     [30, 50, 75])
            eps_per_iter = opt_trial.suggest_int("eps_per_iter",         6,  10)
            abstract_dim = opt_trial.suggest_categorical("abstract_dim", [16, 32, 64])
            width        = opt_trial.suggest_categorical("width",        [64, 128])
            dir_epsilon  = opt_trial.suggest_float("dir_epsilon",        0.05, 0.40)

            print(f"\n[{i:2d}/{N_TRIALS}]  lr={lr:.5f}  c={c:.2f}  d_max={d_max}"
                  f"  sims={num_sims}  eps={eps_per_iter}"
                  f"  dim={abstract_dim}  w={width}  ε={dir_epsilon:.2f}")

            try:
                row, score = run_trial(
                    i, lr, c, d_max, num_sims, eps_per_iter,
                    abstract_dim, width, dir_epsilon,
                    writer, csv_file,
                )
                study.tell(opt_trial, score)
                results.append(row)

                status = "PRUNED" if row["pruned"] else "full "
                print(f"  [{status}] avg={row['agent_avg']:6.2f}  "
                      f"max={row['agent_max']:4d}  "
                      f"pol={row['final_pol_loss']:.4f}  "
                      f"{row['duration_min']:.1f} min")
                if not row["pruned"]:
                    avgs = eval(row["iter_avgs"])
                    step = max(1, len(avgs) // 8)
                    trend = " → ".join(str(v) for v in avgs[::step])
                    print(f"  trend: {trend}")

                best = study.best_trial
                print(f"  Best so far: avg={best.value:.2f}  params={best.params}")

            except Exception as e:
                study.tell(opt_trial, state=optuna.trial.TrialState.FAIL)
                print(f"  ✗ Failed: {e}")
                import traceback; traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    finished = [r for r in results if not r["pruned"]]
    print(f"SUMMARY — top 10 finished trials (of {len(finished)} not pruned)")
    print(f"{'#':>2}  {'lr':>7}  {'c':>4}  {'dmax':>4}  {'sim':>3}  "
          f"{'eps':>3}  {'dim':>3}  {'w':>3}  {'ε':>4}  {'avg':>6}  {'pol':>7}")
    print("-" * 76)
    for r in sorted(finished, key=lambda x: x["agent_avg"], reverse=True)[:10]:
        print(f"{r['trial']:>2}  {r['lr']:>7.5f}  {r['c']:>4.2f}  "
              f"{r['d_max']:>4}  {r['num_sims']:>3}  {r['eps_per_iter']:>3}  "
              f"{r['abstract_dim']:>3}  {r['width']:>3}  {r['dir_epsilon']:>4.2f}  "
              f"{r['agent_avg']:>6.2f}  {r['final_pol_loss']:>7.4f}")

    print("\nOptuna best (including seeded priors):")
    best = study.best_trial
    print(f"  value={best.value:.2f}  params={best.params}")

    pruned_count = sum(1 for r in results if r["pruned"])
    saved_min = pruned_count * (FULL_ITERS - PRUNE_AT) * 2
    print(f"\nPruned: {pruned_count}/{len(results)} trials  "
          f"(saved ~{saved_min} min)")
    print(f"Results → {csv_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
