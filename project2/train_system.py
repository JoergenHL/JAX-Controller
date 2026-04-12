#!/usr/bin/env python3
"""Stage 4B: u-MCTS — self-play with NNr + NNd + NNp, search in abstract space."""

import jax
import jax.numpy as jnp

import config 
from game.LineWorld import LineWorld
from nn.NNManager import NNManager
from rlm import ReinforcementLearningManager


def show_predictions(rlm):
    """Print the network's value and policy for every non-terminal state."""
    print(f"\n  {'State':>6} | {'Value':>7} | {'P(LEFT)':>8} | {'P(RIGHT)':>9}")
    print(f"  {'------':>6}-+-{'-------':>7}-+-{'--------':>8}-+-{'---------':>9}")
    for s in range(1-(config.game["max_position"]), config.game["max_position"]):
        value, policy_logits = rlm._predict(s)
        probs = jax.nn.softmax(jnp.array(policy_logits))
        note = "  <- one step from win" if s == 4 else ("  <- one step from loss" if s == -4 else "")
        print(f"  {s:>6} | {value:>+7.3f} | {float(probs[0]):>8.3f} | {float(probs[1]):>9.3f}{note}")


game = LineWorld()
nnm  = NNManager()
nnm.create_net("nnr", config.nn["nnr_dims"])   # representation: real_state → abstract_state
nnm.create_net("nnd", config.nn["nnd_dims"])   # dynamics:       (abstract_state, action) → (next_abstract, reward)
nnm.create_net("nnp", config.nn["nnp_dims"])   # prediction:     abstract_state → (value, policy)
rlm  = ReinforcementLearningManager(game, nnm)

# ── Before training ────────────────────────────────────────────────────────────
print("=" * 62)
print("BEFORE TRAINING  (random weights)")
print("=" * 62)
show_predictions(rlm)
#win_before = rlm.evaluate(num_games=20)

# ── Training ───────────────────────────────────────────────────────────────────
n_iter = config.training["num_iterations"]
n_ep   = config.training["episodes_per_iter"]
n_ep_it  = config.training["epochs_per_iter"]
print("\n" + "=" * 62)
print(f"TRAINING  ({n_iter} iterations x {n_ep} self-play episodes x {n_ep_it} epochs)")
print("=" * 62)
rlm.train()
win_after = rlm.evaluate(num_games=20)

# ── After training ─────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("AFTER TRAINING")
print("=" * 62)
show_predictions(rlm)


print(f"\n  Win rate: {win_after:.0f}%")

""" # ── Educational explanation ────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("WHAT CHANGED IN STAGE 4B")
print("=" * 62) """

""" STAGE 3:
  real_state ──NNr──> σ ──NNp──> [value, policy]
  MCTS used real game states; GSM drove node expansion.

STAGE 4A:
  Added NNd + full BPTT roll-ahead training.
  MCTS still called GSM to expand nodes.

STAGE 4B (this stage) — u-MCTS:
  MCTS now operates entirely in abstract space.
  The real game is NEVER called during search.

  SEARCH:
    s_k ──NNr──> σ_root
    for each simulation:
      select leaf L via PUCT (using NNp policy priors)
      expand: child σ = NNd(σ_L, a)  for each action a
      evaluate: leaf value = NNp(σ_leaf).value
      backpropagate value to root

  TRAINING (unchanged from Stage 4A — BPTT roll-ahead):
    s_k ──NNr──> σ_k
                  |──NNp──> (v̂_k,  π̂_k)   loss vs (v*_k,  π*_k, -)
                  |──NNd──> (σ_k+1, r̂_k+1) loss vs (-,     -,    r*_k+1)
                              |──NNp──> (v̂_k+1, π̂_k+1) ...
                              ...

  WHY THIS IS THE KEY MUZERO STEP:
    The agent now plans entirely from its own mental model (NNd),
    not from the real game. It never needs to know the rules.
    As NNd improves, so does search quality. As search improves,
    episodes are better quality, which improves all three networks.
    This feedback loop is the core of MuZero.

  WHAT IS STILL MISSING FOR FULL MUZERO:
    Look-back window: currently NNr encodes only s_k (one real state).
    Full MuZero uses the last q+1 real states as input to NNr so the
    abstract state can represent recent history (velocity, momentum, etc.).

 """