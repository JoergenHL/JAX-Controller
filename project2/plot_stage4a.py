#!/usr/bin/env python3
"""Stage 4A visualization: NNd predictions + BPTT training loss."""

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config
from game.LineWorld import LineWorld
from game.ASM import ASM
from nn.NNManager import NNManager
from rlm import ReinforcementLearningManager

# ── Train ──────────────────────────────────────────────────────────────────────
game = LineWorld()
nnm  = NNManager()
nnm.create_net("nnr", config.nn["nnr_dims"])
nnm.create_net("nnd", config.nn["nnd_dims"])
nnm.create_net("nnp", config.nn["nnp_dims"])
rlm  = ReinforcementLearningManager(game, nnm)

result = rlm.train()
losses          = result['losses']
iter_boundaries = result['iter_boundaries']

# ── Collect NNd predictions ────────────────────────────────────────────────────
asm          = ASM()
nn_r         = nnm.get_net("nnr")
nn_d         = nnm.get_net("nnd")
nn_p         = nnm.get_net("nnp")
action_space = game.legal_actions(game.initial_state())   # ["LEFT", "RIGHT"]
states       = list(range(-4, 5))

reward_left  = []
reward_right = []
values       = []
p_right      = []

for s in states:
    sigma = asm.map_abstract_state(s, nn_r)
    _, r_l = asm.next_abstract_state(sigma, "LEFT",  nn_d, action_space)
    _, r_r = asm.next_abstract_state(sigma, "RIGHT", nn_d, action_space)
    reward_left.append(r_l)
    reward_right.append(r_r)

    v, logits = asm.predict(sigma, nn_p)
    probs = jax.nn.softmax(jnp.array(logits))
    values.append(v)
    p_right.append(float(probs[1]))

state_labels = [str(s) for s in states]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
fig.suptitle("Stage 4A — NNd (dynamics) + BPTT roll-ahead training",
             fontsize=13, fontweight='bold')

gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
ax_rew   = fig.add_subplot(gs[0, 0])
ax_val   = fig.add_subplot(gs[0, 1])
ax_pol   = fig.add_subplot(gs[0, 2])
ax_loss  = fig.add_subplot(gs[1, :])

# ── Panel 1: Predicted rewards ─────────────────────────────────────────────────
x = np.arange(len(states))
width = 0.35
ax_rew.bar(x - width/2, reward_left,  width, label='P(reward | LEFT)',  color='#e67e22', alpha=0.85)
ax_rew.bar(x + width/2, reward_right, width, label='P(reward | RIGHT)', color='#3498db', alpha=0.85)
ax_rew.axhline(0, color='black', linewidth=0.6)
ax_rew.set_title("NNd: predicted immediate reward", fontsize=10)
ax_rew.set_xlabel("Real state", fontsize=9)
ax_rew.set_ylabel("Predicted reward", fontsize=9)
ax_rew.set_xticks(x)
ax_rew.set_xticklabels(state_labels, fontsize=8)
ax_rew.legend(fontsize=8)
ax_rew.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# Annotate: state 4 going RIGHT should predict ~+1
ax_rew.annotate("should be\n≈ +1", xy=(x[-1] + width/2, reward_right[-1]),
                xytext=(x[-1] - 0.5, max(reward_right) * 0.7 + 0.1),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=7, color='gray')

# ── Panel 2: Value predictions ─────────────────────────────────────────────────
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]
ax_val.barh(state_labels, values, color=colors, edgecolor='gray', linewidth=0.5)
ax_val.axvline(0, color='black', linewidth=0.8)
ax_val.set_title("NNp: predicted value per state", fontsize=10)
ax_val.set_xlabel("Predicted value", fontsize=9)
ax_val.set_ylabel("Real state", fontsize=9)
ax_val.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

# ── Panel 3: Policy (P(RIGHT)) ─────────────────────────────────────────────────
p_left = [1.0 - pr for pr in p_right]
ax_pol.barh(state_labels, p_left,   color='#e67e22', label='P(LEFT)')
ax_pol.barh(state_labels, p_right,  left=p_left, color='#3498db', label='P(RIGHT)')
ax_pol.axvline(0.5, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax_pol.set_title("NNp: policy per state", fontsize=10)
ax_pol.set_xlabel("Probability", fontsize=9)
ax_pol.set_ylabel("Real state", fontsize=9)
ax_pol.set_xlim(0, 1)
ax_pol.legend(fontsize=8, loc='lower right')

# ── Panel 4: BPTT training loss (4 components) ────────────────────────────────
epochs_axis = list(range(1, len(losses) + 1))
total_loss  = [l[0] for l in losses]
value_loss  = [l[1] for l in losses]
policy_loss = [l[2] for l in losses]
reward_loss = [l[3] for l in losses]

ax_loss.plot(epochs_axis, total_loss,  color='#2196F3', linewidth=1.5, label='Total loss')
ax_loss.plot(epochs_axis, value_loss,  color='#E91E63', linewidth=1.2, label='Value loss')
ax_loss.plot(epochs_axis, policy_loss, color='#4CAF50', linewidth=1.2, label='Policy loss')
ax_loss.plot(epochs_axis, reward_loss, color='#FF9800', linewidth=1.2, label='Reward loss (NEW)')

for i, b in enumerate(iter_boundaries):
    ax_loss.axvline(x=b + 1, color='gray', linestyle='--', linewidth=0.8, alpha=0.6,
                    label="New iteration" if i == 0 else None)
    ax_loss.text(b + 1.5, max(total_loss) * 0.95, f"iter {i+1}", fontsize=7, color='gray', va='top')

ax_loss.set_title("BPTT loss (NNr → NNd^3 → NNp) — reward prediction is the new component",
                  fontsize=10, loc='left')
ax_loss.set_xlabel("Training epoch (across all iterations)", fontsize=9)
ax_loss.set_ylabel("Loss", fontsize=9)
ax_loss.legend(fontsize=8, loc='upper right')
ax_loss.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax_loss.grid(axis='y', alpha=0.3)

out_path = Path(__file__).parent / "stage4a_results.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {out_path}")
plt.show()
