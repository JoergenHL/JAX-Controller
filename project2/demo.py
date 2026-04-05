"""
Demonstration of the modular system with MCTS, Neural Network, and Game State Manager.
Shows:
1. Creating a neural network via NNManager
2. MCTS without neural network (pure rollout)
3. MCTS with neural network (guided by value function)
4. Comparing both approaches
"""

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
    """Demo 1: Creating a neural network via NNManager"""
    print_header("DEMO 1: Creating Neural Network with NNManager")
    
    print("\nConfig-driven architecture:")
    print("- Architecture defined in config.py")
    print("- NNManager loads config and creates model")
    print("- No hardcoded layer dimensions")
    
    nnManager = NNManager()
    model = nnManager.create_model(din=1, dout=1)
    
    print_subheader("Model created successfully")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of layers: {len(model.layers)}")
    print(f"Architecture: Input(1) → Hidden{model.hidden_dims} → Output(1)")
    
    # Test forward pass
    test_input = jnp.ones((3, 1))
    output = model(test_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output.flatten()}")


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
    mcts = MCTS(game, nn_pred=None)  # No NN, pure rollout
    
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
        action = mcts.search(state)
        next_state = game.next_state(state, action)
        reward = game.reward(state, action, next_state)
        total_reward += reward
        path.append(next_state)
        
        print(f"\nStep {step}: MCTS → {action}")
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
    nn_model = nnManager.create_model(din=1, dout=1)
    
    # Create a simple value function from the NN
    def value_fn(state):
        """Use NN to estimate value of a state."""
        state_input = jnp.array([[state]], dtype=jnp.float32)
        value = nn_model(state_input)
        # Return best action based on value (simplified)
        return "RIGHT" if value[0, 0] > 0 else "LEFT"
    
    mcts = MCTS(game, nn_pred=value_fn)
    
    state = game.initial_state()
    print(f"\nSetup: MCTS + Neural Network")
    print(f"NN Architecture: [1] → {nn_model.hidden_dims} → [1]")
    print(f"Simulations per move: {mcts.num_simulations}")
    print("Strategy: NN-guided rollouts during simulation phase")
    
    print_subheader("Initial state")
    print(f"Position: {state}")
    game.render(state)
    
    total_reward = 0
    step = 0
    path = [state]
    
    print_subheader("Running game...")
    while not game.is_terminal(state) and step < 10:
        step += 1
        action = mcts.search(state)
        next_state = game.next_state(state, action)
        reward = game.reward(state, action, next_state)
        total_reward += reward
        path.append(next_state)
        
        print(f"\nStep {step}: MCTS+NN → {action}")
        game.render(next_state)
        print(f"  Reward: {reward:+.1f} | Total: {total_reward:+.1f}")
        
        state = next_state
    
    result = "✓ WIN" if state == game.max_position else "✗ LOSE"
    print(f"\nPath: {' → '.join(map(str, path))}")
    print(f"Result: {result} | Total Reward: {total_reward:+.1f}")


def demo_comparison():
    """Demo 5: Compare MCTS with and without NN over multiple games."""
    print_header("DEMO 5: Comparing MCTS vs MCTS+NN (5 games each)")
    
    game = LineWorld()
    nnManager = NNManager()
    nn_model = nnManager.create_model(din=1, dout=1)
    
    def value_fn(state):
        state_input = jnp.array([[state]], dtype=jnp.float32)
        value = nn_model(state_input)
        return "RIGHT" if value[0, 0] > 0 else "LEFT"
    
    results = {"mcts_only": [], "mcts_nn": []}
    
    # Test MCTS without NN
    print_subheader("Testing pure MCTS...")
    mcts_only = MCTS(game, nn_pred=None)
    
    for i in range(5):
        state = game.initial_state()
        total_reward = 0
        
        while not game.is_terminal(state):
            action = mcts_only.search(state)
            next_state = game.next_state(state, action)
            reward = game.reward(state, action, next_state)
            total_reward += reward
            state = next_state
        
        results["mcts_only"].append(total_reward)
        status = "✓" if state == game.max_position else "✗"
        print(f"  Game {i+1}: {status} Reward: {total_reward:+.1f}")
    
    # Test MCTS with NN
    print_subheader("Testing MCTS + NN...")
    mcts_nn = MCTS(game, nn_pred=value_fn)
    
    for i in range(5):
        state = game.initial_state()
        total_reward = 0
        
        while not game.is_terminal(state):
            action = mcts_nn.search(state)
            next_state = game.next_state(state, action)
            reward = game.reward(state, action, next_state)
            total_reward += reward
            state = next_state
        
        results["mcts_nn"].append(total_reward)
        status = "✓" if state == game.max_position else "✗"
        print(f"  Game {i+1}: {status} Reward: {total_reward:+.1f}")
    
    # Summary
    print_subheader("Summary")
    avg_mcts = sum(results["mcts_only"]) / len(results["mcts_only"])
    avg_mcts_nn = sum(results["mcts_nn"]) / len(results["mcts_nn"])
    wins_mcts = sum(1 for r in results["mcts_only"] if r > 0)
    wins_mcts_nn = sum(1 for r in results["mcts_nn"] if r > 0)
    
    print(f"\nMCTS Only:")
    print(f"  Win rate: {wins_mcts}/5 ({wins_mcts*20}%)")
    print(f"  Avg reward: {avg_mcts:+.2f}")
    
    print(f"\nMCTS + NN:")
    print(f"  Win rate: {wins_mcts_nn}/5 ({wins_mcts_nn*20}%)")
    print(f"  Avg reward: {avg_mcts_nn:+.2f}")
    
    if avg_mcts != 0:
        improvement = ((avg_mcts_nn - avg_mcts) / abs(avg_mcts) * 100)
        print(f"\nImprovement: {improvement:+.1f}%")


def demo_architecture():
    """Demo 6: Show the modular architecture."""
    print_header("DEMO 6: Modular System Architecture")
    
    print("""
The system is built with modularity in mind:

1. GAME STATE MANAGER (LineWorld)
   - Defines game rules, transitions, rewards
   - Completely independent of MCTS and NN
   - Can be swapped for any game with compatible interface

2. NEURAL NETWORK MANAGER (NNManager)
   - Reads architecture from config.py
   - Creates models dynamically
   - Decoupled from MCTS - can train separately

3. MCTS
   - Pure search algorithm
   - Works with OR WITHOUT neural network
   - nn_pred parameter is optional (defaults to None)
   
4. INTEGRATION (rlm.py)
   - Brings all components together
   - Wires config → NNManager → MCTS → Game
   - Easy to add training loops, logging, etc.

Benefits:
✓ Easy to test each component independently
✓ Easy to swap components (different game, different NN)
✓ Config drives behavior (architecture, hyperparameters)
✓ Clear separation of concerns
✓ Can use NN alone for supervised learning
✓ Can use MCTS alone for pure search
✓ Can combine them for hybrid approach
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
