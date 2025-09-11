import numpy as np
import random
from collections import defaultdict

from rlnexus.core.policy import Policy

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
