from game.LineWorld import LineWorld

def test_lineworld():
    """Complete test suite for LineWorld game."""
    
    game = LineWorld()
    
    print("=" * 50)
    print("LINEWORLD TEST SUITE")
    print("=" * 50)
    
    # Test 1: Initial State
    print("\n[TEST 1] Initial State")
    initial = game.initial_state()
    print(f"Initial state: {initial}")
    assert initial == 0, "Initial state should be 0"
    print("✓ PASSED")
    
    # Test 2: Legal Actions at different states
    print("\n[TEST 2] Legal Actions")
    print(f"Legal actions at state 0: {game.legal_actions(0)}")
    assert game.legal_actions(0) == ["LEFT", "RIGHT"], "Should have both actions"
    print(f"Legal actions at state 3 (terminal): {game.legal_actions(3)}")
    assert game.legal_actions(3) == [], "Terminal state should have no legal actions"
    print(f"Legal actions at state -3 (terminal): {game.legal_actions(-3)}")
    assert game.legal_actions(-3) == [], "Terminal state should have no legal actions"
    print("✓ PASSED")
    
    # Test 3: Next State Transitions
    print("\n[TEST 3] Next State Transitions")
    state = 0
    print(f"From state {state}:")
    next_right = game.next_state(state, "RIGHT")
    print(f"  Move RIGHT → {next_right}")
    assert next_right == 1, "RIGHT should increment state"
    next_left = game.next_state(state, "LEFT")
    print(f"  Move LEFT → {next_left}")
    assert next_left == -1, "LEFT should decrement state"
    print("✓ PASSED")
    
    # Test 4: Reward Calculation
    print("\n[TEST 4] Reward Calculation")
    # Reaching max_position (3)
    reward_win = game.reward(2, "RIGHT", 3)
    print(f"Reward for reaching position 3: {reward_win}")
    assert reward_win == 1, "Should get +1 for reaching position 3"
    # Reaching min position (-3)
    reward_lose = game.reward(-2, "LEFT", -3)
    print(f"Reward for reaching position -3: {reward_lose}")
    assert reward_lose == -1, "Should get -1 for reaching position -3"
    # Regular movement
    reward_normal = game.reward(0, "RIGHT", 1)
    print(f"Reward for normal move: {reward_normal}")
    assert reward_normal == 0, "Should get 0 for regular movement"
    print("✓ PASSED")
    
    # Test 5: Terminal State Detection
    print("\n[TEST 5] Terminal State Detection")
    print(f"Is state 0 terminal? {game.is_terminal(0)}")
    assert not game.is_terminal(0), "State 0 should not be terminal"
    print(f"Is state 3 terminal? {game.is_terminal(3)}")
    assert game.is_terminal(3), "State 3 should be terminal"
    print(f"Is state -3 terminal? {game.is_terminal(-3)}")
    assert game.is_terminal(-3), "State -3 should be terminal"
    print(f"Is state 2 terminal? {game.is_terminal(2)}")
    assert not game.is_terminal(2), "State 2 should not be terminal"
    print("✓ PASSED")
    
    # Test 6: Render
    print("\n[TEST 6] Render Display")
    print("Rendering state 0:")
    game.render(0)
    print("Rendering state 1:")
    game.render(1)
    print("Rendering state -2:")
    game.render(-2)
    print("✓ PASSED")
    
    # Test 7: Simulate a complete game to terminal state (moving RIGHT)
    print("\n[TEST 7] Game Simulation - Moving RIGHT to win")
    state = game.initial_state()
    total_reward = 0
    step = 0
    print(f"Step {step}: State = {state}")
    game.render(state)
    
    while not game.is_terminal(state):
        action = "RIGHT"
        next_state = game.next_state(state, action)
        reward = game.reward(state, action, next_state)
        total_reward += reward
        step += 1
        
        print(f"Step {step}: Action = {action}, State = {next_state}, Reward = {reward}, Total = {total_reward}")
        game.render(next_state)
        state = next_state
    
    print(f"Game ended at state {state} with total reward: {total_reward}")
    assert total_reward == 1, "Total reward should be 1 for reaching position 3"
    print("✓ PASSED")
    
    # Test 8: Simulate a complete game to terminal state (moving LEFT)
    print("\n[TEST 8] Game Simulation - Moving LEFT to lose")
    state = game.initial_state()
    total_reward = 0
    step = 0
    print(f"Step {step}: State = {state}")
    game.render(state)
    
    while not game.is_terminal(state):
        action = "LEFT"
        next_state = game.next_state(state, action)
        reward = game.reward(state, action, next_state)
        total_reward += reward
        step += 1
        
        print(f"Step {step}: Action = {action}, State = {next_state}, Reward = {reward}, Total = {total_reward}")
        game.render(next_state)
        state = next_state
    
    print(f"Game ended at state {state} with total reward: {total_reward}")
    assert total_reward == -1, "Total reward should be -1 for reaching position -3"
    print("✓ PASSED")
    
    # Test 9: Mixed actions
    print("\n[TEST 9] Mixed Actions Sequence")
    state = game.initial_state()
    actions_sequence = ["RIGHT", "RIGHT", "LEFT", "RIGHT", "RIGHT"]
    total_reward = 0
    print(f"Initial: State = {state}")
    game.render(state)
    
    for i, action in enumerate(actions_sequence, 1):
        if game.is_terminal(state):
            print(f"Game already terminal at state {state}")
            break
        next_state = game.next_state(state, action)
        reward = game.reward(state, action, next_state)
        total_reward += reward
        print(f"Step {i}: Action = {action}, State = {next_state}, Reward = {reward}, Total = {total_reward}, Terminal = {game.is_terminal(next_state)}")
        game.render(next_state)
        state = next_state
    
    print("✓ PASSED")
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✓")
    print("=" * 50)

if __name__ == "__main__":
    test_lineworld()
