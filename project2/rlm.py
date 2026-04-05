from flax import nnx
import jax.numpy as jnp
import jax

import config
from game.LineWorld import LineWorld
from mcts.mcts import MCTS
from nn.NNManager import NNManager

gsm = LineWorld()
nnManager = NNManager()
nn_pred = nnManager.create_model(1, 1)
mcts = MCTS(gsm, nn_pred)

state = gsm.initial_state()
total_reward = 0

while not gsm.is_terminal(state):
    action = mcts.search(state)

    next_state = gsm.next_state(state, action)
    reward = gsm.reward(state, action, next_state)
    
    total_reward += reward
    state = next_state

EH = []

for episode in range(config.rlm["episodes"]):
    epidata = []

    state = gsm.initial_state()

    for steps in range(config.rlm["steps"]):
        pass

print(f"Game ended with total reward: {total_reward}")


