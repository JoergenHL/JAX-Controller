from abc import ABC, abstractmethod

class IController(ABC):

    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, error):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def update_params(self, params):
        pass