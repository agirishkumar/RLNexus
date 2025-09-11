import numpy as np
from rlnexus.core.mdp import ModelBasedEnvironment

class GridWorld(ModelBasedEnvironment):
    """
    A simple GridWorld environment.
    The agent starts at a specified location and must navigate to a goal.
    """
    def __init__(self, size=5, start=(0, 0), goal=None, obstacles=None, rewards=None):
        self.size = size
        self.start_pos = start
        self.goal_pos = goal if goal else (size - 1, size - 1)
        self.obstacles = set(obstacles) if obstacles else set()
        
        default_rewards = {'goal': 1.0, 'obstacle': -1.0, 'step': -0.1}
        self.rewards = default_rewards
        if rewards:
            self.rewards.update(rewards)

        self.state = None
        self.action_space = [0, 1, 2, 3] # 0:Up, 1:Down, 2:Left, 3:Right
        self.observation_space = list(np.ndindex((size, size)))

        self.reset()

    @property
    def model(self):
        # Generate the model on the fly
        m = {}
        for state in self.observation_space:
            m[state] = {}
            for action in self.action_space:
                next_s, reward, done, _ = self._get_next_state_info(state, action)
                m[state][action] = [(1.0, next_s, reward, done)]
        return m

    def _get_next_state_info(self, state, action):
        if state in self.obstacles or state == self.goal_pos:
            return state, 0, True, {}

        x, y = state
        if action == 0: # Up
            x = max(0, x - 1)
        elif action == 1: # Down
            x = min(self.size - 1, x + 1)
        elif action == 2: # Left
            y = max(0, y - 1)
        elif action == 3: # Right
            y = min(self.size - 1, y + 1)
        
        next_state = (x, y)

        if next_state == self.goal_pos:
            reward = self.rewards['goal']
            done = True
        elif next_state in self.obstacles:
            reward = self.rewards['obstacle']
            done = True
        else:
            reward = self.rewards['step']
            done = False
        
        return next_state, reward, done, {}

    def step(self, action):
        next_state, reward, done, info = self._get_next_state_info(self.state, action)
        self.state = next_state
        return self.state, reward, done, info

    def reset(self):
        self.state = self.start_pos
        return self.state

    def render(self, mode='human'):
        grid = np.full((self.size, self.size), '_', dtype=str)
        grid[self.goal_pos] = 'G'
        for obs in self.obstacles:
            grid[obs] = 'X'
        grid[self.state] = 'A'
        
        print("\n".join(" ".join(row) for row in grid))
        print("-" * self.size * 2)
