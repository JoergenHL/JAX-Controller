class LineWorld:

    def __init__(self):
        self.max_position = 5
        self.reward_right = 1
        self.reward_left = -1
        self.state_dim    = 1
        self.num_actions  = 2
        self.reward_scale = 1.0   # rewards are ±1; no scaling needed
    

    def initial_state(self):
        return 0
    

    @property
    def action_space(self):
        """All possible actions regardless of current state."""
        return ["LEFT", "RIGHT"]

    def legal_actions(self, state):
        if self.is_terminal(state):
            return []
        else:
            return ["LEFT", "RIGHT"]
    

    def next_state(self, state, action):
        if action == "LEFT":
            return state - 1
        else:
            return state + 1
    

    def reward(self, state, action, next_state):
        if next_state == self.max_position:
            return self.reward_right
        elif next_state == -self.max_position:
            return self.reward_left
        else:
            return 0

    def is_terminal(self, state):
        return abs(state) == self.max_position

    def is_win(self, state):
        return state == self.max_position

    def max_tile(self, state):
        """Return the current position (used by evaluate for progress reporting)."""
        return int(state)
    
    def render(self, state):
            positions = list(range(-self.max_position, self.max_position + 1))

            # Print the line
            line = ""
            for p in positions:
                line += f"{p:>3} "
            print(line)

            # Print pointer
            pointer = ""
            for p in positions:
                if p == state:
                    pointer += "  ^ "
                else:
                    pointer += "    "
            print(pointer)
            print()
    

    
    

