#!/usr/bin/env python3
"""Stage 4B: u-MCTS — self-play with NNr + NNd + NNp, search in abstract space."""

import config
from game.TwentyFortyEight import TwentyFortyEight
from nn.NNManager import NNManager
from rlm import ReinforcementLearningManager
from baseline import RandomBaseline
from run_logger import RunLogger
from visualize import plot_training, replay_game

# ── Guard required for multiprocessing spawn mode ──────────────────────────────
# With spawn (macOS/Windows default), worker processes re-import this file to
# reconstruct the module context. Without this guard, every worker would run
# the training code again and try to spawn its own workers — causing a crash.
# All executable code must live inside this block.
if __name__ == "__main__":

    # ── Network setup ──────────────────────────────────────────────────────────
    # Dimensions are derived from the game object + config hyperparameters.
    # To swap games: change the import above and nothing else.
    # To change network architecture: edit config.nn["nnr_hidden"] etc.

    game = TwentyFortyEight()
    s = game.state_dim    # input size for NNr  (1 for LineWorld, 16 for 2048)
    a = game.num_actions  # action count         (2 for LineWorld,  4 for 2048)
    d = config.nn["abstract_dim"]

    nnr_dims = [s]     + config.nn["nnr_hidden"] + [d    ]
    nnp_dims = [d]     + config.nn["nnp_hidden"] + [1 + a]
    nnd_dims = [d + a] + config.nn["nnd_hidden"] + [d + 1]

    nnm = NNManager()
    nnm.create_net("nnr", nnr_dims)
    nnm.create_net("nnd", nnd_dims)
    nnm.create_net("nnp", nnp_dims)

    rlm = ReinforcementLearningManager(game, nnm)

    # ── Training ───────────────────────────────────────────────────────────────
    n_iter   = config.training["num_iterations"]
    n_ep     = config.training["episodes_per_iter"]
    n_epochs = config.training["epochs_per_iter"]
    print("=" * 62)
    print(f"TRAINING  ({n_iter} iter × {n_ep} episodes × {n_epochs} epochs/iter)")
    print(f"  NNr: {nnr_dims}")
    print(f"  NNp: {nnp_dims}")
    print(f"  NNd: {nnd_dims}")
    print("=" * 62)

    result = rlm.train()

    # ── Final evaluation ───────────────────────────────────────────────────────
    pct, avg_tile, tiles = rlm.evaluate(num_games=10)

    # ── Logging & plotting ─────────────────────────────────────────────────────
    network_dims = {"nnr": nnr_dims, "nnp": nnp_dims, "nnd": nnd_dims}
    logger = RunLogger(game, config, network_dims)
    logger.log_run(result)
    logger.log_eval(pct, avg_tile, tiles)
    json_path = logger.save()

    # ── Model checkpoint ───────────────────────────────────────────────────────
    model_path = json_path.replace(".json", ".pkl")
    logger.data["model_path"] = model_path
    json_path = logger.save()   # re-save JSON with model_path recorded
    nnm.save(model_path)

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
    )

    # ── Game replay ───────────────────────────────────────────────────────────
    if config.viz.get("replay_after_training", False):
        replay_game(rlm, max_steps=config.viz.get("replay_max_steps", 50))
