from abc import ABC, abstractmethod

class IGameStateManager():
    
    @abstractmethod
    def __init__():
        raise NotImplementedError    
    
    @abstractmethod
    def initial_state(self):
        raise NotImplementedError    
    
    @abstractmethod
    def legal_actions(self, state):
        raise NotImplementedError
    
    @abstractmethod
    def next_state(self, state, action):
        raise NotImplementedError
    
    @abstractmethod
    def reward(self, state, action, next_state):
        raise NotImplementedError
    
    @abstractmethod
    def is_terminal(self, state):
        raise NotImplementedError
    
