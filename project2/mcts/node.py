

class Node:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action

        self.children = {}

        self.visits = 0

        self.action_stats = {}

    def is_expanded(self):
        return len(self.children) > 0 
    
    def update(self, action, value):
        stats = self.action_stats[action]

        stats["N"] += 1
        stats["W"] += value
        stats["Q"] = stats["W"] / stats["N"]

        self.visits += 1

    def add_child(self, action, child_node):
        self.children[action] = child_node
        self.action_stats[action] = {"N": 0, "W": 0, "Q": 0}


