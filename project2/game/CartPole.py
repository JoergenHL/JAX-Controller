"""CartPole game — stateless, analytic implementation.

The gym CartPole environment is stateful (env.step mutates internal state),
which is incompatible with this project's pure-function GSM interface where
next_state(state, action) must be a deterministic pure function.

This file re-implements the CartPole physics analytically so every transition
is a pure function of (state, action). The dynamics match gym CartPole-v1
exactly (same constants, Euler integration, same terminal bounds).

State: numpy float32 array of shape (4,)
    [x, x_dot, theta, theta_dot]
    x         — cart position (m)
    x_dot     — cart velocity (m/s)
    theta     — pole angle (rad, 0 = upright)
    theta_dot — pole angular velocity (rad/s)

Actions: "LEFT" or "RIGHT" (force applied to cart)

Reward: +1 for every non-terminal step — perfectly aligned with survival.
    A better agent survives longer → higher cumulative return → higher value.
    The value network has a clean linear target: expected steps remaining.

Terminal: pole angle > 12° OR cart position > ±2.4 m.
    500-step cap is enforced by the episode loop (max_steps), not is_terminal.

reward_scale: 50. A random agent survives ~10 steps (return ~10); a trained
    agent survives 500 (return 500). Dividing by 50 keeps value targets in
    the 0.2–10 range — well-conditioned for MSE.
"""

import math
import numpy as np


class CartPole:

    # Physics constants matching gym CartPole-v1
    GRAVITY        = 9.8
    MASS_CART      = 1.0
    MASS_POLE      = 0.1
    TOTAL_MASS     = 1.1          # MASS_CART + MASS_POLE
    HALF_POLE_LEN  = 0.5          # half the pole length
    POLE_MASS_LEN  = 0.05         # MASS_POLE * HALF_POLE_LEN
    FORCE_MAG      = 10.0
    TAU            = 0.02         # seconds per step (Euler integration)

    # Terminal bounds
    X_THRESHOLD     = 2.4
    THETA_THRESHOLD = 12 * math.pi / 180   # 12 degrees in radians

    ACTIONS = ["LEFT", "RIGHT"]

    def __init__(self):
        self.state_dim    = 4
        self.num_actions  = 2
        self.reward_scale = 50.0
        self.score_label  = "Steps survived"

    @property
    def action_space(self):
        return self.ACTIONS

    def initial_state(self):
        """Uniform random start in [-0.05, 0.05] for all state variables.

        Matches gym CartPole-v1 reset() exactly.
        """
        return np.random.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)

    def legal_actions(self, state):
        """Both actions are always legal while the episode is alive."""
        if self.is_terminal(state):
            return []
        return self.ACTIONS

    def next_state(self, state, action):
        """One Euler step of CartPole physics.

        Pure function: does not mutate any object state.
        Matches gym CartPole-v1 step() kinematics exactly.
        """
        x, x_dot, theta, theta_dot = state

        force = self.FORCE_MAG if action == "RIGHT" else -self.FORCE_MAG

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Standard CartPole equations of motion
        temp = (force + self.POLE_MASS_LEN * theta_dot ** 2 * sin_theta) / self.TOTAL_MASS
        theta_acc = (
            (self.GRAVITY * sin_theta - cos_theta * temp)
            / (self.HALF_POLE_LEN * (4.0 / 3.0 - self.MASS_POLE * cos_theta ** 2 / self.TOTAL_MASS))
        )
        x_acc = temp - self.POLE_MASS_LEN * theta_acc * cos_theta / self.TOTAL_MASS

        # Euler integration
        x         = x         + self.TAU * x_dot
        x_dot     = x_dot     + self.TAU * x_acc
        theta     = theta     + self.TAU * theta_dot
        theta_dot = theta_dot + self.TAU * theta_acc

        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def reward(self, state, action, next_state):
        """+ 1 for every step the pole stays up.

        Reward fires on the transition *into* next_state (before terminal check),
        so a 500-step game earns exactly 500 total reward.
        """
        return 1.0

    def is_terminal(self, state):
        """Episode over when pole falls or cart leaves the track."""
        x, _, theta, _ = state
        return (abs(x) > self.X_THRESHOLD or abs(theta) > self.THETA_THRESHOLD)

    def is_win(self, state):
        """CartPole has no explicit win — report False (win rate not meaningful here)."""
        return False

    def max_tile(self, state):
        """Return 1 while pole is upright, 0 when fallen."""
        return 0 if self.is_terminal(state) else 1

    def eval_score(self, steps: int, final_state) -> int:
        """Evaluation score for one episode: steps survived (= total reward)."""
        return steps

    def render(self, state):
        """ASCII visualisation: show cart position and pole angle."""
        x, _, theta, _ = state
        bar_width = 40
        center = bar_width // 2
        cart_pos = int(center + (x / self.X_THRESHOLD) * center)
        cart_pos = max(0, min(bar_width - 1, cart_pos))

        bar = ["-"] * bar_width
        bar[cart_pos] = "C"
        angle_deg = math.degrees(theta)
        print(f"|{''.join(bar)}|  angle={angle_deg:+.1f}°  x={x:+.2f}")
