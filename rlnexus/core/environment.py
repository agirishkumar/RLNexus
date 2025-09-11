from abc import ABC, abstractmethod
from typing import Any, Tuple

class Environment(ABC):
    """
    Abstract base class for all reinforcement learning environments.
    """
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            action: The action to be taken by the agent.
            
        Returns:
            A tuple containing (state, reward, done, info).
        """
        pass

    @abstractmethod
    def reset(self) -> Any:
        """
        Resets the environment to an initial state.
        
        Returns:
            The initial state of the environment.
        """
        pass

    @abstractmethod
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment.
        """
        pass
