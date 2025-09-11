import numpy as np
from rlnexus.core.environment import Environment

class GridWorld(Environment):
    """
    A simple GridWorld environment.
    The agent starts at a specified location and must navigate to a goal.
    """
    def __init__(self, size=5, start=(0, 0), goal=None, obstacles=None):
        self.size = size
        self.start_pos = start
        self.goal_pos = goal if goal else (size - 1, size - 1)
        self.obstacles = set(obstacles) if obstacles else set()

        self.state = None
        self.action_space = [0, 1, 2, 3] # 0:Up, 1:Down, 2:Left, 3:Right
        self.observation_space = (size, size)

        self.reset()

    def step(self, action):
        if self.state in self.obstacles or self.state == self.goal_pos:
            return self.state, 0, True, {}

        x, y = self.state
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
            reward = 1.0
            done = True
        elif next_state in self.obstacles:
            reward = -1.0
            done = True
        else:
            reward = -0.1
            done = False
        
        self.state = next_state
        return self.state, reward, done, {}

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
