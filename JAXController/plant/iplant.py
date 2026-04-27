from abc import ABC, abstractmethod

class IPlant(ABC):
    """ 
    Plant interface
    """

    @abstractmethod
    def init_state(self):
        """ 
        Return inital plant state 
        """
        pass
    
    @abstractmethod
    def step(self, state, U, D):
        """ 
        Compute one timestep

        Return next state 
        """
        pass

    @abstractmethod
    def output(self, state):
        """ 
        Compute plant output Y from current state
        """
        pass