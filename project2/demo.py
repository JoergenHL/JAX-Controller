"""
Demonstration of the modular system with MCTS, Neural Network, and Game State Manager.
Shows:
1. Creating a neural network via NNManager
2. MCTS without neural network (pure rollout)
3. MCTS with neural network (guided by value function)
4. Comparing both approaches
"""

import jax
import jax.numpy as jnp
from game.LineWorld import LineWorld
from mcts.mcts import MCTS
from nn.NNManager import NNManager


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title):
    """Print a formatted subheader."""
    print(f"\n>>> {title}")


def demo_neural_network_creation():
    """Demo 1: Creating and composing networks"""
    print_header("DEMO 1: Simple, Modular Networks")
    
    print("\nCreate networks by specifying dimensions:")
    print("- dims = [input, hidden1, hidden2, ..., output]")
    print("- Compose networks by using outputs as inputs")
    print("- No magic: you control the flow")
    
    nnManager = NNManager()
    
    # Create simple networks - one per component
    trunk = nnManager.create_net("trunk", [1, 16, 16])
    value_head = nnManager.create_net("value", [16, 1])
    policy_head = nnManager.create_net("policy", [16, 2])
    
    print_subheader("Networks created")
    print(f"Available: {list(nnManager.models.keys())}")
    print(f"  trunk: state(1) → hidden(16) → hidden(16)")
    print(f"  value: hidden(16) → value(1)")
    print(f"  policy: hidden(16) → policy(2)")
    
    # Compose: state → trunk → value and policy
    print_subheader("Forward pass (state = 0.5)")
    import jax.numpy as jnp
    state = jnp.array([[0.5]], dtype=jnp.float32)
    
    trunk_out = trunk(state)
    value = value_head(trunk_out)
    policy_logits = policy_head(trunk_out)
    policy_probs = jax.nn.softmax(policy_logits[0])
    
    print(f"State: {state.flatten()}")
    print(f"  Trunk output shape: {trunk_out.shape}")
    print(f"  Value: {value[0, 0]:.3f}")
    print(f"  Policy logits: {policy_logits[0]}")
    print(f"  Policy probs: {policy_probs}")
    
    # Show you can also create other networks independently
    print_subheader("Other network types (same way)")
    repr_net = nnManager.create_net("repr", [1, 8])  # obs → latent
    dyn_net = nnManager.create_net("dyn", [8+2, 8+1])  # (latent,action) → (next_latent,reward)
    
    test_obs = jnp.array([[0.3]], dtype=jnp.float32)
    latent = repr_net(test_obs)
    print(f"Representation: obs(0.3) → latent shape {latent.shape}")
    
    action_concat = jnp.concatenate([latent, jnp.array([[1.0, 0.0]])], axis=1)  # latent + one-hot
    dyn_out = dyn_net(action_concat)
    print(f"Dynamics: (latent, action) → output shape {dyn_out.shape}")
    print(f"  Next latent: {dyn_out[0, :8]}")
    print(f"  Reward: {dyn_out[0, 8]:.3f}")


def demo_basic_game():
    """Demo 2: Basic game without AI."""
    print_header("DEMO 2: Basic LineWorld Game (No AI)")
    
    game = LineWorld()
    print("\nGame Rules:")
    print("- Start position: 0")
    print("- Reach +3 → WIN (reward +1)")
    print("- Reach -3 → LOSE (reward -1)")
    print("- Actions: LEFT (-1) or RIGHT (+1)")
    
    state = game.initial_state()
    print_subheader("Manual play: Move RIGHT 3 times")
    print(f"Initial: {state}")
    game.render(state)
    
    total_reward = 0
    path = [state]
    
    for step, action in enumerate(["RIGHT", "RIGHT", "RIGHT"], 1):
        next_state = game.next_state(state, action)
        reward = game.reward(state, action, next_state)
        total_reward += reward
        path.append(next_state)
        
        print(f"\nStep {step}: {action}")
        game.render(next_state)
        print(f"  Reward: {reward:+.1f} | Total: {total_reward:+.1f}")
        state = next_state
        if game.is_terminal(state):
            break
    
    result = "✓ WIN" if state == game.max_position else "✗ LOSE"
    print(f"\nPath: {' → '.join(map(str, path))}")
    print(f"Result: {result} | Final Reward: {total_reward:+.1f}")


def demo_mcts_without_nn():
    """Demo 3: MCTS without neural network guidance."""
    print_header("DEMO 3: MCTS Without Neural Network")
    
    game = LineWorld()
    mcts = MCTS(game, num_actions=2, nn_pred=None)  # No NN, pure rollout
    
    state = game.initial_state()
    print(f"\nSetup: MCTS with {mcts.num_simulations} simulations per move")
    print("Strategy: Random rollouts during simulation phase")
    
    print_subheader("Initial state")
    print(f"Position: {state}")
    game.render(state)
    
    total_reward = 0
    step = 0
    path = [state]
    
    print_subheader("Running game...")
    while not game.is_terminal(state) and step < 10:
        step += 1
        action, policy, value = mcts.search(state)
        next_state = game.next_state(state, action)
        reward = game.reward(state, action, next_state)
        total_reward += reward
        path.append(next_state)
        
        print(f"\nStep {step}: MCTS → {action} (value={value:+.3f})")
        game.render(next_state)
        print(f"  Reward: {reward:+.1f} | Total: {total_reward:+.1f}")
        
        state = next_state
    
    result = "✓ WIN" if state == game.max_position else "✗ LOSE"
    print(f"\nPath: {' → '.join(map(str, path))}")
    print(f"Result: {result} | Total Reward: {total_reward:+.1f}")


def demo_mcts_with_nn():
    """Demo 4: MCTS with neural network guidance."""
    print_header("DEMO 4: MCTS With Neural Network")
    
    game = LineWorld()
    nnManager = NNManager()
    
    # Create simple networks for value/policy
    trunk = nnManager.create_net("trunk", [1, 16, 16])
    value_head = nnManager.create_net("value", [16, 1])
    policy_head = nnManager.create_net("policy", [16, 2])
    
    # Simple prediction function: forward pass through all heads
    def nn_pred(state):
        state_batch = jnp.array([[state]], dtype=jnp.float32)
        trunk_out = trunk(state_batch)
        value = float(value_head(trunk_out)[0, 0])
        policy_logits = policy_head(trunk_out)[0]
        policy_probs = jax.nn.softmax(policy_logits)
        policy_dict = {i: float(policy_probs[i]) for i in range(len(policy_probs))}
        return value, policy_dict
    
    # DEBUG: Print NN predictions for all states
    print_subheader("DEBUG: NN predictions for different states")
    print("State | Value | Policy (LEFT, RIGHT)")
    print("-" * 50)
    for s in [-3, -2, -1, 0, 1, 2, 3]:
        v, p = nn_pred(float(s))
        print(f"{s:5d} | {v:+8.2f} | L:{p[0]:.3f}  R:{p[1]:.3f}")
    print("-" * 50)
    mcts = MCTS(game, num_actions=2, nn_pred=nn_pred)
    
    state = game.initial_state()
    print(f"\nSetup: MCTS + Multi-head Network")
    print(f"  Trunk: state(1) → hidden(16) → hidden(16)")
    print(f"  Value head: hidden(16) → value(1)")
    print(f"  Policy head: hidden(16) → policy(2)")
    print(f"Simulations per move: {mcts.num_simulations}")
    
    print_subheader("Initial state")
    print(f"Position: {state}")
    game.render(state)
    
    total_reward = 0
    step = 0
    path = [state]
    
    print_subheader("Running game...")
    while not game.is_terminal(state) and step < 10:
        step += 1
        action, policy, value = mcts.search(state)
        next_state = game.next_state(state, action)
        reward = game.reward(state, action, next_state)
        total_reward += reward
        path.append(next_state)
        
        print(f"\nStep {step}: MCTS+NN → {action} (value={value:+.3f})")
        game.render(next_state)
        print(f"  Reward: {reward:+.1f} | Total: {total_reward:+.1f}")
        
        state = next_state
    
    result = "✓ WIN" if state == game.max_position else "✗ LOSE"
    print(f"\nPath: {' → '.join(map(str, path))}")
    print(f"Result: {result} | Total Reward: {total_reward:+.1f}")


def demo_comparison():
    """Demo 5: Compare MCTS with and without NN over multiple games."""
    print_header("DEMO 5: Comparing MCTS vs MCTS+NN (20 games each for reliability)")
    
    results = {"mcts_only": [], "mcts_nn": []}
    
    game = LineWorld()
    nnManager = NNManager()
    
    # Create simple networks
    trunk = nnManager.create_net("trunk", [1, 16, 16])
    value_head = nnManager.create_net("value", [16, 1])
    policy_head = nnManager.create_net("policy", [16, 2])
    
    # Track NN calls
    nn_call_count = [0]
    
    def nn_pred(state):
        nn_call_count[0] += 1
        state_batch = jnp.array([[state]], dtype=jnp.float32)
        trunk_out = trunk(state_batch)
        value = float(value_head(trunk_out)[0, 0])
        policy_logits = policy_head(trunk_out)[0]
        policy_probs = jax.nn.softmax(policy_logits)
        policy_dict = {i: float(policy_probs[i]) for i in range(len(policy_probs))}
        return value, policy_dict
    
    # First, show what MCTS decides for state 0 with and without NN
    print_subheader("Single move analysis at state 0")
    
    mcts_only = MCTS(game, num_actions=2, nn_pred=None)
    mcts_nn = MCTS(game, num_actions=2, nn_pred=nn_pred)
    
    print("\nPure MCTS (10 sims) from state 0:")
    action1, policy1, value1 = mcts_only.search(0, debug=True)
    print(f"  Best action: {action1}, Policy: {policy1}, Value: {value1:+.3f}")
    
    print("\nMCTS+NN (10 sims) from state 0:")
    action2, policy2, value2 = mcts_nn.search(0, debug=True)
    print(f"  Best action: {action2}, Policy: {policy2}, Value: {value2:+.3f}")
    
    print("-" * 50)
    print_subheader("Testing pure MCTS (20 games)...")
    mcts_only = MCTS(game, num_actions=2, nn_pred=None)
    
    for i in range(20):
        state = game.initial_state()
        total_reward = 0
        
        while not game.is_terminal(state):
            action, _, _ = mcts_only.search(state)
            next_state = game.next_state(state, action)
            reward = game.reward(state, action, next_state)
            total_reward += reward
            state = next_state
        
        results["mcts_only"].append(total_reward)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/20 games completed")
    
    # Test MCTS with NN
    print_subheader("Testing MCTS + NN (20 games)...")
    nn_call_count[0] = 0
    mcts_nn = MCTS(game, num_actions=2, nn_pred=nn_pred)
    
    for i in range(20):
        state = game.initial_state()
        total_reward = 0
        
        while not game.is_terminal(state):
            action, _, _ = mcts_nn.search(state)
            next_state = game.next_state(state, action)
            reward = game.reward(state, action, next_state)
            total_reward += reward
            state = next_state
        
        results["mcts_nn"].append(total_reward)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/20 games completed")
    
    # Summary
    print_subheader("Summary (20 games each)")
    avg_mcts = sum(results["mcts_only"]) / len(results["mcts_only"])
    avg_mcts_nn = sum(results["mcts_nn"]) / len(results["mcts_nn"])
    wins_mcts = sum(1 for r in results["mcts_only"] if r > 0)
    wins_mcts_nn = sum(1 for r in results["mcts_nn"] if r > 0)
    
    print(f"\nMCTS Only:")
    print(f"  Win rate: {wins_mcts}/20 ({wins_mcts*5}%)")
    print(f"  Avg reward: {avg_mcts:+.2f}")
    
    print(f"\nMCTS + NN (untrained):")
    print(f"  Win rate: {wins_mcts_nn}/20 ({wins_mcts_nn*5}%)")
    print(f"  Avg reward: {avg_mcts_nn:+.2f}")
    print(f"  NN evaluations: {nn_call_count[0]}")
    
    if avg_mcts != 0:
        improvement = ((avg_mcts_nn - avg_mcts) / abs(avg_mcts) * 100)
        print(f"\nImprovement: {improvement:+.1f}%")


def demo_architecture():
    """Demo 6: Show the modular architecture."""
    print_header("DEMO 6: Simple, Composable Architecture")
    
    print("""
Simple principles:

1. GAME: LineWorld
   - Pure state transitions
   - No coupling to learning/search

2. NETWORKS: Just MLPs
   - MLP(dims) = specify [input, hidden1, hidden2, ..., output]
   - No magic, no special heads built-in
   - Compose into any structure you want

3. NNManager: Simple factory
   - create_net(name, dims) → MLP
   - get_net(name) → retrieve
   - That's it. No prediction logic.

4. MCTS: Pure search
   - Optional nn_pred function: state → (value, policy_dict)
   - You provide the function, MCTS uses it
   - Works with OR WITHOUT NN

Composability Examples:

Single NN for value+policy:
  net = nnManager.create_net("net", [1, 16, 16, 3])
  output = net(state)
  value, policy_logits = output[:, :1], output[:, 1:]

Multi-head (trunk + heads):
  trunk = nnManager.create_net("trunk", [1, 16, 16])
  value = nnManager.create_net("value", [16, 1])
  policy = nnManager.create_net("policy", [16, 2])
  x = trunk(state)
  v, p = value(x), policy(x)

MuZero (representation + dynamics + prediction):
  repr = nnManager.create_net("repr", [obs_dim, latent_dim])
  dyn = nnManager.create_net("dyn", [latent_dim+num_actions, latent_dim+1])
  pred = nnManager.create_net("pred", [latent_dim, 1+num_actions])

Key: No hidden complexity. What you see is what you get.
Every network is just an MLP. Compose them however you need.
    """)



if __name__ == "__main__":
    print("\n" + "📊 " * 20)
    print("    MODULAR AI SYSTEM DEMONSTRATION")
    print("📊 " * 20)
    
    # Run all demos
    demo_neural_network_creation()
    demo_basic_game()
    demo_mcts_without_nn()
    demo_mcts_with_nn()
    demo_comparison()
    demo_architecture()
    
    print("\n" + "=" * 70)
    print("  Demo completed!")
    print("=" * 70 + "\n")
