# AI programming projects

Two self-contained projects from coursework on AI programming.

## [JAXController/](JAXController/README.md)

Gradient-based learning of control systems in JAX. PID and neural-net controllers learn to drive different plants (bathtub water level, Cournot competition, LIF neuron) by backpropagating through a differentiable forward simulation.

## [MuZero/](MuZero/README.md)

A from-scratch [MuZero](https://arxiv.org/abs/1911.08265) implementation that learns to play **2048** without being told the rules. Three small MLPs (representation, dynamics, prediction) are trained jointly via BPTT; planning happens with u-MCTS in the learned latent space.

