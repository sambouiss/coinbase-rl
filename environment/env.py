
import datetime
from abc import ABC, abstractmethod
from typing import List

State = List[float]
Action = List[float]

class Environment(ABC):
    """
    Environment manages the RL environment. It holds components needed to: set up, query and interact with the enviroment.

    This could be written as a GYM environment but I wanted to keep everything self contained - mainly because its an edifying experience.

    It would probably be better to make it a GYM enviroment because you'd get access to all the frameworks written for GYM enviroments already.
    """

    @abstractmethod
    def setupEnv(self, *args) ->  List[State]:
        """
        setupEnv starts the enviorment and returns the initial state. Currently the state is 2-D to potenially
        allow for multi-agent RL. 
        """
        pass

    @abstractmethod
    def stepEnv(self, *args) -> List[State]:
        """
        stepEnv updates the information and 
        """
        pass

    @abstractmethod
    def act(self, actions: List[Action], *args) -> None:
        """
        Given a list of actions, act implements the logic to run those actions. 
        """
        pass
       

class ProductFeatures(ABC):
    """
    ProductFeatures contains all the information relavent to a product for the enviroment.

    ProductFeatures manages:
        - updating the products information
        - turning the information into a state
        - acting on the product
    """
        
    @abstractmethod
    def update(self, *args) -> State:
        """
        Update implements the logic to update state.
        """
        return self.getProductFeatures()
        
    @abstractmethod
    def getProductFeatures(self, *args) -> State:
        """
        Get product features returns the state representation.
        """
        pass
    
    @abstractmethod
    def act(self, action: Action) -> None:
        """
        Act takes an action and implements the logic to interact with the enviroment.
        I represented actions as bid and ask offsets, but there could be other action representations.
        """
        pass