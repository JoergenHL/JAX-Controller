#!/usr/bin/env python3
"""Tests for Stage 3: NNr + NNp pipeline through ASM."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import config
from game.LineWorld import LineWorld
from game.ASM import ASM
from nn.NNManager import NNManager
from rlm import ReinforcementLearningManager


def make_system():
    """Create a fresh game + NNM + RLM for testing."""
    game = LineWorld()
    nnm  = NNManager()
    nnm.create_net("nnr", config.nn["nnr_dims"])
    nnm.create_net("nnp", config.nn["nnp_dims"])
    rlm  = ReinforcementLearningManager(game, nnm)
    return game, nnm, rlm


# ── Test 1 ─────────────────────────────────────────────────────────────────────

def test_abstract_states_differ():
    """NNr should map different real states to different abstract vectors."""
    _, nnm, rlm = make_system()
    nn_r = nnm.get_net("nnr")

    abstracts = [rlm.asm.map_abstract_state(s, nn_r) for s in range(-4, 5)]
    for i in range(len(abstracts)):
        for j in range(i + 1, len(abstracts)):
            diff = float(jnp.max(jnp.abs(abstracts[i] - abstracts[j])))
            assert diff > 1e-6, (
                f"States {i-4} and {j-4} map to identical abstract states — "
                "NNr is not differentiating inputs."
            )
    print("PASS  test_abstract_states_differ")


# ── Test 2 ─────────────────────────────────────────────────────────────────────

def test_prediction_structure():
    """ASM.predict should return (scalar float, array of length num_actions)."""
    _, nnm, rlm = make_system()
    nn_r = nnm.get_net("nnr")
    nn_p = nnm.get_net("nnp")

    abstract = rlm.asm.map_abstract_state(0, nn_r)
    value, policy_logits = rlm.asm.predict(abstract, nn_p)

    assert isinstance(value, float), f"Value should be float, got {type(value)}"
    assert len(policy_logits) == 2, (
        f"Expected 2 policy logits (LEFT, RIGHT), got {len(policy_logits)}"
    )
    print("PASS  test_prediction_structure")


# ── Test 3 ─────────────────────────────────────────────────────────────────────

def test_joint_training_updates_both_nets():
    """One training step must update weights in BOTH NNr and NNp.

    This verifies that gradients flow all the way back through NNp into NNr.
    If NNr weights don't change, the gradient chain is broken.
    """
    _, nnm, _ = make_system()

    def first_weight(net):
        """Read the first weight matrix of a network as a numpy array."""
        return np.array(net.layers[0].w.value)

    # Capture weight snapshots before training
    w_nnr_before = first_weight(nnm.get_net("nnr"))
    w_nnp_before = first_weight(nnm.get_net("nnp"))

    # Minimal fake training data
    states         = [0.0, 1.0, 2.0]
    value_targets  = [1.0, 1.0, 1.0]
    policy_targets = [[0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]

    nnm.train_repr_pred(states, value_targets, policy_targets, num_epochs=1)

    w_nnr_after = first_weight(nnm.get_net("nnr"))
    w_nnp_after = first_weight(nnm.get_net("nnp"))

    assert not np.allclose(w_nnr_before, w_nnr_after, atol=1e-10), (
        "NNr weights did not change — gradients are NOT flowing back through NNr. "
        "Check that loss_fn in train_repr_pred chains both networks."
    )
    assert not np.allclose(w_nnp_before, w_nnp_after, atol=1e-10), (
        "NNp weights did not change after training step."
    )
    print("PASS  test_joint_training_updates_both_nets")


# ── Test 4 ─────────────────────────────────────────────────────────────────────

def test_win_rate_after_training():
    """Win rate should reach at least 80% after the full training loop."""
    _, _, rlm = make_system()
    rlm.train()
    win_pct = rlm.evaluate(num_games=100)
    assert win_pct >= 80, (
        f"Win rate {win_pct:.0f}% is below 80% — training may not be converging."
    )
    print(f"PASS  test_win_rate_after_training  ({win_pct:.0f}% wins)")


# ── Test 5 ─────────────────────────────────────────────────────────────────────

def test_asm_pipeline_consistent_with_direct_predict():
    """rlm._predict(s) and manual ASM calls should return identical results."""
    _, nnm, rlm = make_system()
    nn_r = nnm.get_net("nnr")
    nn_p = nnm.get_net("nnp")

    for s in [-3, 0, 3]:
        v1, p1 = rlm._predict(s)
        abstract = rlm.asm.map_abstract_state(s, nn_r)
        v2, p2   = rlm.asm.predict(abstract, nn_p)
        assert abs(v1 - v2) < 1e-5, f"Value mismatch for state {s}: {v1} vs {v2}"
        diff = float(jnp.max(jnp.abs(jnp.array(p1) - jnp.array(p2))))
        assert diff < 1e-5, f"Policy mismatch for state {s}"
    print("PASS  test_asm_pipeline_consistent_with_direct_predict")


# ── Run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Stage 3 tests")
    print("=" * 55)

    test_abstract_states_differ()
    test_prediction_structure()
    test_joint_training_updates_both_nets()
    test_asm_pipeline_consistent_with_direct_predict()
    test_win_rate_after_training()   # slow — runs full training loop

    print("=" * 55)
    print("All tests passed.")
    print("=" * 55)
