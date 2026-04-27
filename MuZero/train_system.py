#!/usr/bin/env python3
"""Stage 4B: u-MCTS — self-play with NNr + NNd + NNp, search in abstract space."""

import json
import os
import shutil
from datetime import datetime

import config
from nn.NNManager import NNManager
from rlm import ReinforcementLearningManager
from baseline import RandomBaseline
from run_logger import RunLogger
from visualize import plot_training, plot_policy_analysis, replay_game
from worker import _get_game

# ── Guard required for multiprocessing spawn mode ──────────────────────────────
# With spawn (macOS/Windows default), worker processes re-import this file to
# reconstruct the module context. Without this guard, every worker would run
# the training code again and try to spawn its own workers — causing a crash.
# All executable code must live inside this block.
if __name__ == "__main__":

    # ── Game instantiation from config ─────────────────────────────────────────
    # Change config.game["name"] to swap games — no other file needs touching.
    game = _get_game(config.game["name"])
    s = game.state_dim    # input size for NNr  (1 for LineWorld, 4 for CartPole, 16 for 2048)
    a = game.num_actions  # action count         (2 for LineWorld,  4 for 2048)
    d = config.nn["abstract_dim"]

    q        = config.nn.get("q", 0)
    nnr_dims = [s * (q + 1)] + config.nn["nnr_hidden"] + [d]
    nnp_dims = [d]     + config.nn["nnp_hidden"] + [1 + a]
    nnd_dims = [d + a] + config.nn["nnd_hidden"] + [d + 1]

    nnm = NNManager()
    nnm.create_net("nnr", nnr_dims)
    nnm.create_net("nnd", nnd_dims)
    nnm.create_net("nnp", nnp_dims)

    rlm = ReinforcementLearningManager(game, nnm)

    # ── Training ───────────────────────────────────────────────────────────────
    n_iter    = config.training["num_iterations"]
    n_ep      = config.training["episodes_per_iter"]
    n_updates = config.training["updates_per_iter"]
    mbs       = config.training["minibatch_size"]

    # Shared timestamp so logger, interval checkpoints, and leaderboard files
    # all share one prefix — makes runs/ easy to group visually.
    network_dims = {"nnr": nnr_dims, "nnp": nnp_dims, "nnd": nnd_dims}
    run_ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    game_name   = game.__class__.__name__
    runs_dir    = "runs"
    os.makedirs(runs_dir, exist_ok=True)
    run_prefix  = os.path.join(runs_dir, f"{game_name}_{run_ts}")

    print("=" * 62)
    print(f"TRAINING  ({n_iter} iter × {n_ep} episodes × {n_updates} updates/iter, mbs={mbs})")
    print(f"  NNr: {nnr_dims}")
    print(f"  NNp: {nnp_dims}")
    print(f"  NNd: {nnd_dims}")
    print(f"  Run prefix: {run_prefix}")
    print("=" * 62)

    result = rlm.train(
        run_prefix=run_prefix,
        network_dims=network_dims,
        game_name=game_name,
    )

    leaderboard  = result.get("leaderboard", [])

    score_label = getattr(game, "score_label", "Score")

    # ── Final evaluation ───────────────────────────────────────────────────────
    pct, avg_tile, tiles = rlm.evaluate(num_games=100)

    # ── Logging & plotting ─────────────────────────────────────────────────────
    logger = RunLogger(game, config, network_dims, timestamp_str=run_ts)
    logger.log_run(result)
    logger.log_eval(pct, avg_tile, tiles)
    json_path = logger.save()

    # ── Model checkpoint ───────────────────────────────────────────────────────
    model_path = json_path.replace(".json", ".pkl")
    logger.data["model_path"] = model_path
    json_path = logger.save()   # re-save JSON with model_path recorded
    nnm.save(model_path)

    # ── Top-K shootout: re-evaluate leaderboard on many games ──────────────────
    # During training each entry was selected on a noisy 50-game eval. Now
    # play final_eval_games on each and re-rank. Cheap (greedy NNr+NNp only).
    final_eval_games = config.viz.get("final_eval_games", 1000)
    if leaderboard and final_eval_games > 0:
        print("\n" + "=" * 62)
        print(f"  TOP-{len(leaderboard)} SHOOTOUT — {final_eval_games} games per candidate")
        print("=" * 62)

        ranking = []   # list of dicts for logger.log_leaderboard
        for entry in leaderboard:
            print(f"\n  → iter {entry['iteration']:3d}  "
                  f"(selection avg={entry['eval_avg']:.1f})")
            # Load this entry's weights into the live nnm, run evaluate() which
            # already supports parallel pools, then record results.
            nnm.set_layer_weights(entry["layer_weights"])
            _, final_avg, final_scores = rlm.evaluate(num_games=final_eval_games)
            ranking.append({
                "iteration":          entry["iteration"],
                "selection_eval_avg": round(float(entry["eval_avg"]), 2),
                "final_eval_avg":     round(float(final_avg), 2),
                "final_eval_games":   final_eval_games,
                "pkl_path":           entry.get("pkl_path"),
            })

        ranking.sort(key=lambda e: e["final_eval_avg"], reverse=True)
        for rank, entry in enumerate(ranking, start=1):
            entry["rank"] = rank

        # Print a compact ranking table
        print("\n  Final ranking:")
        print(f"  {'rank':>4}  {'iter':>4}  {'selection':>10}  {'final':>8}")
        for e in ranking:
            print(f"  {e['rank']:>4}  {e['iteration']:>4}  "
                  f"{e['selection_eval_avg']:>10.1f}  "
                  f"{e['final_eval_avg']:>8.1f}")

        # Copy winner's pkl + json into a stable "_champion" name for downstream scripts
        winner = ranking[0]
        if winner["pkl_path"] and os.path.exists(winner["pkl_path"]):
            champion_pkl  = f"{run_prefix}_champion.pkl"
            champion_json = f"{run_prefix}_champion.json"
            shutil.copy2(winner["pkl_path"], champion_pkl)
            src_json = winner["pkl_path"].replace(".pkl", ".json")
            if os.path.exists(src_json):
                with open(src_json) as f:
                    champ_meta = json.load(f)
                champ_meta["kind"] = "champion"
                champ_meta["final_eval_avg"] = winner["final_eval_avg"]
                champ_meta["final_eval_games"] = final_eval_games
                with open(champion_json, "w") as f:
                    json.dump(champ_meta, f, indent=2)
            print(f"\n  Champion → {champion_pkl}")

        # Persist ranking in the main run JSON and as a standalone file
        logger.log_leaderboard(ranking)
        json_path = logger.save()
        ranking_path = f"{run_prefix}_final_ranking.json"
        with open(ranking_path, "w") as f:
            json.dump({
                "run_prefix":       run_prefix,
                "final_eval_games": final_eval_games,
                "ranking":          ranking,
            }, f, indent=2)
        print(f"  Ranking saved → {ranking_path}")

        # Restore the final-iter weights into nnm so downstream policy analysis
        # / replay still reflect the end-of-training model, not the champion.
        final_path = json_path.replace(".json", ".pkl")
        if os.path.exists(final_path):
            nnm.load(final_path)

    # ── Baseline comparison ────────────────────────────────────────────────────
    baseline_scores = None
    if config.viz.get("compare_baseline", False):
        baseline_agent = RandomBaseline(game)
        b_pct, b_avg, b_tiles = baseline_agent.evaluate(
            num_games=config.viz.get("baseline_games", 20)
        )
        logger.log_baseline(b_pct, b_avg, b_tiles)
        json_path = logger.save()   # overwrite with baseline data included
        baseline_scores = (b_pct, b_avg, b_tiles)

    # ── MCTS comparison (optional upper bound) ─────────────────────────────────
    mcts_scores = None
    if config.viz.get("compare_mcts", False):
        m_pct, m_avg, m_tiles = rlm.evaluate_mcts(
            num_games=config.viz.get("mcts_eval_games", 5)
        )
        logger.log_mcts_eval(m_pct, m_avg, m_tiles)
        json_path = logger.save()   # overwrite with MCTS data included
        mcts_scores = (m_pct, m_avg, m_tiles)

    plot_training(
        result,
        game_name=game.__class__.__name__,
        network_dims=network_dims,
        save_path=json_path.replace(".json", ".png"),
        baseline=baseline_scores,
        mcts_eval=mcts_scores,
        score_label=score_label,
    )

    # ── Policy analysis ────────────────────────────────────────────────────────
    if config.viz.get("show_policy_analysis", False):
        policy_data = rlm.sample_policy_data(
            num_games=config.viz.get("policy_analysis_games", 20)
        )
        plot_policy_analysis(
            policy_data,
            game_name=game.__class__.__name__,
            save_path=json_path.replace(".json", "_policy.png"),
            score_label=score_label,
        )

    # ── Game replay ───────────────────────────────────────────────────────────
    if config.viz.get("replay_after_training", False):
        replay_game(rlm, max_steps=config.viz.get("replay_max_steps", 50))
