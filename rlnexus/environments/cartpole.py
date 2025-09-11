import numpy as np
from rlnexus.core.environment import Environment
from rlnexus.linalg.core import Tensor

class CartPole(Environment):
    """
    A classic CartPole environment.
    """
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        self.action_space = [0, 1] # 0: Push left, 1: Push right
        self.observation_space_shape = (4,)
        self.state = None

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        else:
            reward = 0.0

        return Tensor(self.state), reward, done, {}

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return Tensor(self.state)

    def render(self, mode='human'):
        # Simple text-based rendering
        x, _, theta, _ = self.state
        cart_pos = int((x + self.x_threshold) / (2 * self.x_threshold) * 40)
        pole_angle = int(theta * 180 / np.pi)
        
        print(f"Cart Position: {' ' * cart_pos}[]")
        print(f"Pole Angle: {pole_angle}°")
