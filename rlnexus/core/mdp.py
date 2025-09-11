from abc import abstractmethod
from rlnexus.core.environment import Environment

class ModelBasedEnvironment(Environment):
    """
    An abstract base class for model-based environments.
    It extends the basic Environment by requiring an explicit model 
    of the environment's dynamics (MDP).
    """
    @property
    @abstractmethod
    def model(self):
        """
        The transition model of the MDP. This should be a nested dictionary-like
        structure: model[state][action] = [(prob, next_state, reward, done), ...].
        """
        pass
