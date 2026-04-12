#!/usr/bin/env python3
"""Tests for Stage 4A: NNd + BPTT roll-ahead training."""

import jax
import jax.numpy as jnp
import numpy as np

import config
from game.LineWorld import LineWorld
from game.ASM import ASM
from nn.NNManager import NNManager
from rlm import ReinforcementLearningManager


def make_system():
    game = LineWorld()
    nnm  = NNManager()
    nnm.create_net("nnr", config.nn["nnr_dims"])
    nnm.create_net("nnd", config.nn["nnd_dims"])
    nnm.create_net("nnp", config.nn["nnp_dims"])
    rlm  = ReinforcementLearningManager(game, nnm)
    return game, nnm, rlm


# ── Test 1 ─────────────────────────────────────────────────────────────────────

def test_nnd_output_shape():
    """NNd must output [abstract_dim + 1] = [5] given [abstract_dim + num_actions] = [6] input."""
    _, nnm, rlm = make_system()
    asm  = ASM()
    nn_r = nnm.get_net("nnr")
    nn_d = nnm.get_net("nnd")
    action_space = ["LEFT", "RIGHT"]

    sigma = asm.map_abstract_state(0, nn_r)                          # [1, 4]
    next_sigma, reward_pred = asm.next_abstract_state(sigma, "RIGHT", nn_d, action_space)

    assert next_sigma.shape == (1, config.nn["abstract_dim"]), (
        f"next_sigma shape {next_sigma.shape} != (1, {config.nn['abstract_dim']})"
    )
    assert isinstance(reward_pred, float), f"reward_pred should be float, got {type(reward_pred)}"
    print("PASS  test_nnd_output_shape")


# ── Test 2 ─────────────────────────────────────────────────────────────────────

def test_different_actions_give_different_next_states():
    """NNd must produce different next abstract states for LEFT vs RIGHT."""
    _, nnm, rlm = make_system()
    asm  = ASM()
    nn_r = nnm.get_net("nnr")
    nn_d = nnm.get_net("nnd")
    action_space = ["LEFT", "RIGHT"]

    sigma = asm.map_abstract_state(0, nn_r)
    sigma_left,  _ = asm.next_abstract_state(sigma, "LEFT",  nn_d, action_space)
    sigma_right, _ = asm.next_abstract_state(sigma, "RIGHT", nn_d, action_space)

    diff = float(jnp.max(jnp.abs(sigma_left - sigma_right)))
    assert diff > 1e-6, (
        "NNd produces identical next states for LEFT and RIGHT — "
        "the action one-hot encoding may not be reaching the network."
    )
    print("PASS  test_different_actions_give_different_next_states")


# ── Test 3 ─────────────────────────────────────────────────────────────────────

def test_bptt_updates_all_three_networks():
    """One BPTT step must update NNr, NNd, AND NNp weights.

    This is the key test: if any network's weights are unchanged, gradients
    are not flowing through that part of the unrolled computation graph.
    """
    _, nnm, rlm = make_system()

    def first_weight(net):
        return np.array(net.layers[0].w.get_value())

    w_r_before = first_weight(nnm.get_net("nnr"))
    w_d_before = first_weight(nnm.get_net("nnd"))
    w_p_before = first_weight(nnm.get_net("nnp"))

    minibatches = [{
        'state':          2.0,   # non-zero so dL/dW = dL/dσ * input != 0
        'action_indices': [1, 1, 1],     # RIGHT, RIGHT, Right
        'value_targets':  [1.0, 1.0, 1.0],
        'policy_targets': [[0.2, 0.8], [0.1, 0.9], [0.0, 1.0]],
        'reward_targets': [0.0, 0.0, 1.0],
    }]
    nnm.train_bptt(minibatches,
                   abstract_dim=config.nn["abstract_dim"],
                   num_actions=config.nn["num_actions"],
                   num_epochs=1)

    w_r_after = first_weight(nnm.get_net("nnr"))
    w_d_after = first_weight(nnm.get_net("nnd"))
    w_p_after = first_weight(nnm.get_net("nnp"))

    assert not np.allclose(w_r_before, w_r_after, atol=1e-10), \
        "NNr weights did not change — gradients not flowing back through NNd→NNr"
    assert not np.allclose(w_d_before, w_d_after, atol=1e-10), \
        "NNd weights did not change"
    assert not np.allclose(w_p_before, w_p_after, atol=1e-10), \
        "NNp weights did not change"
    print("PASS  test_bptt_updates_all_three_networks")


# ── Test 4 ─────────────────────────────────────────────────────────────────────

def test_reward_loss_decreases():
    """Reward loss should decrease over the course of BPTT training.

    Note: NNd is trained on unrolled latent trajectories (NNr → NNd^w), not on
    NNr-encoded states directly. There is no consistency constraint forcing
    NNd(NNr(s), a) to equal NNd(NNd(NNr(s-1), a), a). So checking reward
    prediction from a directly encoded state is not meaningful here.

    What IS meaningful: the reward component of the BPTT loss decreases,
    meaning NNd learns to predict rewards along the unrolled trajectory.
    """
    _, nnm, rlm = make_system()

    result = rlm.train()
    losses = result['losses']

    # losses is a list of (total, value, policy, reward) per epoch.
    # Compare the first epoch of training to the last.
    # We avoid comparing first-N vs last-N *across iterations* because each
    # new iteration adds fresh diverse data to the buffer, which can temporarily
    # raise the reward loss before NNd re-fits the expanded dataset.
    first_reward = losses[0][3]
    last_reward  = losses[-1][3]

    assert last_reward < first_reward, (
        f"Reward loss did not decrease: epoch 1={first_reward:.4f}, "
        f"final epoch={last_reward:.4f}. NNd may not be learning."
    )
    print(f"PASS  test_reward_loss_decreases  "
          f"(reward loss: {first_reward:.4f} → {last_reward:.4f})")


# ── Test 5 ─────────────────────────────────────────────────────────────────────

def test_win_rate_after_training():
    """Win rate should remain >= 80% with BPTT training."""
    _, _, rlm = make_system()
    rlm.train()
    win_pct = rlm.evaluate(num_games=30)
    assert win_pct >= 80, f"Win rate {win_pct:.0f}% is below 80%"
    print(f"PASS  test_win_rate_after_training  ({win_pct:.0f}% wins)")


# ── Run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Stage 4A tests")
    print("=" * 55)

    test_nnd_output_shape()
    test_different_actions_give_different_next_states()
    test_bptt_updates_all_three_networks()
    test_reward_loss_decreases()    # runs full training loop
    test_win_rate_after_training()    # runs full training loop

    print("=" * 55)
    print("All tests passed.")
    print("=" * 55)
