import numpy as np
import random
from collections import defaultdict

from rlnexus.core.policy import Policy
from rlnexus.linalg.core import Tensor

class EpsilonGreedyPolicy(Policy):
    """
    An epsilon-greedy policy for tabular methods.
    It selects the best action with probability 1-epsilon,
    and a random action with probability epsilon.
    """
    def __init__(self, q_table, epsilon, action_space):
        self.q_table = q_table
        self.epsilon = epsilon
        self.action_space = action_space

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            q_values = self.q_table[state]
            # If all q-values for a state are the same, choose randomly
            if len(set(q_values)) == 1:
                return random.choice(self.action_space)
            return np.argmax(q_values)


class StochasticPolicy(Policy):
    """
    A stochastic policy implemented as a neural network.
    Outputs action probabilities using softmax.
    """
    def __init__(self, network):
        self.network = network

    def select_action(self, state):
        """
        Sample an action from the policy's probability distribution.
        """
        if not isinstance(state, Tensor):
            state = Tensor(state)
        if state.data.ndim == 1:
            state.data = state.data.reshape(1, -1)

        # Get action logits from network
        logits = self.network(state)

        # Convert to probabilities
        probs = logits.softmax()

        # Sample action from probability distribution
        action_probs = probs.data.flatten()
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def get_action_probabilities(self, state):
        """
        Get the probability distribution over actions for a given state.
        """
        if not isinstance(state, Tensor):
            state = Tensor(state)
        if state.data.ndim == 1:
            state.data = state.data.reshape(1, -1)

        logits = self.network(state)
        probs = logits.softmax()
        return probs.data.flatten()
