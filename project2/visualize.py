"""Game-agnostic visualization and replay utilities.

plot_training(result, ...)  — 3-panel PNG: loss curves, eval scores, loss breakdown
replay_game(rlm, ...)       — greedy game replay printed to stdout
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Loss + eval plot ───────────────────────────────────────────────────────────

def plot_training(result: dict, game_name: str, network_dims: dict,
                  save_path: str = None, baseline=None, mcts_eval=None):
    """Save a 3-panel training summary figure.

    Panels:
      1. Loss curves (total / value / policy / reward) over all epochs.
         Vertical dashed lines mark iteration boundaries.
      2. Eval scores over training iterations.
         Each checkpoint shows all per-game scores as dots, with lines for
         the average and the best game, so variance is visible.
         If baseline is provided, a dashed gray reference line + shaded band
         shows the random-agent score range for comparison.
         Skipped if no eval data (eval_every=0) AND no baseline provided.
      3. Per-iteration loss breakdown: stacked bars of value / policy / reward
         at the final epoch of each iteration — shows which component dominates.

    Args:
        result:       dict returned by rlm.train()
        game_name:    string label for the title (e.g. "TwentyFortyEight")
        network_dims: dict with 'nnr', 'nnp', 'nnd' dim lists for subtitle
        save_path:    if provided, save to this path; otherwise show interactively
        baseline:     optional (pct, avg_tile, max_tiles) from RandomBaseline.evaluate()
    """
    losses          = result["losses"]
    boundaries      = result["iter_boundaries"]
    eval_scores     = result.get("eval_scores", [])

    has_eval     = len(eval_scores) > 0
    has_baseline = baseline is not None
    has_mcts     = mcts_eval is not None
    n_panels = 3 if (has_eval or has_baseline or has_mcts) else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: Loss curves ──────────────────────────────────────────────────
    ax = axes[0]
    if losses:
        epochs    = list(range(1, len(losses) + 1))
        total_l   = [l[0] for l in losses]
        value_l   = [l[1] for l in losses]
        policy_l  = [l[2] for l in losses]
        reward_l  = [l[3] for l in losses]

        ax.plot(epochs, total_l,  label="total",  color="black",   linewidth=1.5)
        ax.plot(epochs, value_l,  label="value",  color="tab:blue",  linewidth=1.0)
        ax.plot(epochs, policy_l, label="policy", color="tab:orange", linewidth=1.0)
        ax.plot(epochs, reward_l, label="reward", color="tab:green",  linewidth=1.0)

        for b in boundaries[1:]:
            ax.axvline(b + 1, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training loss")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    else:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Training loss")

    # ── Panel 2: Eval scores (+ optional baseline / MCTS reference) ──────────
    if has_eval or has_baseline or has_mcts:
        ax = axes[1]

        if has_eval:
            iters   = [e[0] for e in eval_scores]
            avgs    = [e[2] for e in eval_scores]
            bests   = [max(e[3]) for e in eval_scores]
            all_pts = [(e[0], t) for e in eval_scores for t in e[3]]

            xs, ys = zip(*all_pts) if all_pts else ([], [])
            ax.scatter(xs, ys, color="tab:blue", alpha=0.4, s=20, label="per-game")
            ax.plot(iters, avgs,  color="tab:blue",   linewidth=1.5, marker="o",
                    markersize=4, label="avg")
            ax.plot(iters, bests, color="tab:orange", linewidth=1.5, marker="s",
                    markersize=4, label="best", linestyle="--")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        else:
            # No in-training eval — show a placeholder note
            ax.text(0.5, 0.6, "No in-training eval\n(set eval_every > 0)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")

        # Baseline reference: dashed line at avg, shaded band from min to max
        if has_baseline:
            _, b_avg, b_tiles = baseline
            b_min, b_max = min(b_tiles), max(b_tiles)
            # Use x-axis span of the eval data, or a default span of [0.5, 1.5]
            x_lo = 0.5
            x_hi = max(iters) + 0.5 if has_eval else 1.5
            ax.axhline(b_avg, color="gray", linestyle="--", linewidth=1.2,
                       label=f"random avg ({b_avg:.0f})")
            ax.fill_between([x_lo, x_hi], b_min, b_max,
                            color="gray", alpha=0.12, label="random range")

        # MCTS reference: dashed purple line at avg
        if has_mcts:
            _, m_avg, m_tiles = mcts_eval
            x_lo = 0.5
            x_hi = max(iters) + 0.5 if has_eval else 1.5
            ax.axhline(m_avg, color="tab:purple", linestyle="--", linewidth=1.2,
                       label=f"MCTS avg ({m_avg:.0f})")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Max tile")
        ax.set_title("Eval: max tile per game")
        ax.legend(fontsize=8)

    # ── Panel 3: Loss breakdown per iteration ─────────────────────────────────
    ax = axes[-1]
    if losses and boundaries:
        num_iters = len(boundaries)
        iter_labels = list(range(1, num_iters + 1))
        final_v, final_p, final_r = [], [], []

        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < num_iters else len(losses)
            last = losses[end - 1] if end > start else losses[start]
            final_v.append(last[1])
            final_p.append(last[2])
            final_r.append(last[3])

        x = np.arange(len(iter_labels))
        w = 0.6
        ax.bar(x, final_v, w, label="value",  color="tab:blue")
        ax.bar(x, final_p, w, bottom=final_v, label="policy", color="tab:orange")
        bottom_rp = [v + p for v, p in zip(final_v, final_p)]
        ax.bar(x, final_r, w, bottom=bottom_rp, label="reward", color="tab:green")

        ax.set_xticks(x)
        ax.set_xticklabels(iter_labels)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (final epoch)")
        ax.set_title("Loss breakdown per iteration")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Loss breakdown")

    # ── Figure title ──────────────────────────────────────────────────────────
    dim_str = (f"NNr {network_dims.get('nnr', '?')}  "
               f"NNp {network_dims.get('nnp', '?')}  "
               f"NNd {network_dims.get('nnd', '?')}")
    fig.suptitle(f"{game_name} — {dim_str}", fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ── Game replay ────────────────────────────────────────────────────────────────

def replay_game(rlm, max_steps: int = 50):
    """Play one greedy episode and render each board state to stdout.

    Uses argmax of MCTS visit counts (no sampling) so the policy is
    deterministic. Stops at terminal or max_steps.

    Args:
        rlm:       ReinforcementLearningManager with trained networks
        max_steps: safety limit on episode length
    """
    gsm   = rlm.gsm
    state = gsm.initial_state()
    total_reward = 0.0

    print(f"\n{'─' * 40}")
    print(f"  Greedy replay  (max {max_steps} steps)")
    print(f"{'─' * 40}")
    print("Initial board:")
    gsm.render(state)

    for step in range(1, max_steps + 1):
        if gsm.is_terminal(state):
            break

        _, policy, _ = rlm.mcts.search(state)
        # Greedy: pick the action with the most visits
        action = max(policy, key=policy.get)

        next_state = gsm.next_state(state, action)
        reward     = gsm.reward(state, action, next_state)
        total_reward += reward

        print(f"Step {step:3d} | action={action:<6} | reward={reward:6.1f} "
              f"| max tile={gsm.max_tile(next_state)}")
        gsm.render(next_state)

        state = next_state

    final_tile = gsm.max_tile(state)
    print(f"{'─' * 40}")
    print(f"  Episode ended  steps={step}  total_reward={total_reward:.1f}"
          f"  max_tile={final_tile}")
    print(f"{'─' * 40}\n")
