class LineWorld:

    def __init__(self):
        self.max_position = 5
        self.reward_right = 1
        self.reward_left = -1
    

    def initial_state(self):
        return 0
    

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
    

    
    

