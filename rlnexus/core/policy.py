from abc import ABC, abstractmethod
from typing import Any

class Policy(ABC):
    """
    Abstract base class for all policies.
    A policy defines the agent's behavior by mapping states to actions.
    """
    @abstractmethod
    def select_action(self, state: Any) -> Any:
        """
        Select an action based on the current state.
        
        Args:
            state: The current state of the environment.
            
        Returns:
            The action to be taken by the agent.
        """
        pass
