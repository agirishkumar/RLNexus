import numpy as np
from collections import defaultdict

from rlnexus.core.agent import Agent
from rlnexus.core.environment import Environment
from rlnexus.core.policies import EpsilonGreedyPolicy

class SARSAAgent(Agent):
    """
    A SARSA agent for tabular environments.
    """
    def __init__(self, env: Environment, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(len(env.action_space)))
        policy = EpsilonGreedyPolicy(self.q_table, epsilon, env.action_space)
        super().__init__(policy, env)
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def act(self, state):
        return self.policy.select_action(state)

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        SARSA's learn method requires the next action to be passed in.
        """
        old_value = self.q_table[state][action]
        next_q_value = self.q_table[next_state][next_action]
        
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_q_value - old_value)
        
        self.q_table[state][action] = new_value
