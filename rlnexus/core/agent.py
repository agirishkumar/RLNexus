from abc import ABC, abstractmethod
from typing import Any, Tuple
from rlnexus.core.policy import Policy
from rlnexus.core.environment import Environment

class Agent(ABC):
    """
    Abstract base class for all reinforcement learning agents.
    """
    def __init__(self, policy: Policy, env: Environment):
        self.policy = policy
        self.env = env

    @abstractmethod
    def act(self, state: Any) -> Any:
        """
        Choose an action based on the current state.
        
        Args:
            state: The current state of the environment.
            
        Returns:
            The action to be taken.
        """
        pass

    @abstractmethod
    def learn(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """
        Update the agent's knowledge based on an experience tuple.
        
        Args:
            state: The state before the action was taken.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action was taken.
            done: Whether the episode has terminated.
        """
        pass
