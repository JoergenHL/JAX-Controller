from abc import ABC, abstractmethod

class IController(ABC):
    
    @abstractmethod
    def step(self, error):
        pass

    @abstractmethod
    def get_params(self):
        pass
