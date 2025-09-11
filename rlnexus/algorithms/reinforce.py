import numpy as np
from rlnexus.core.agent import Agent
from rlnexus.core.environment import Environment
from rlnexus.core.policies import StochasticPolicy
from rlnexus.linalg.core import Tensor
from rlnexus.linalg.nn import SGD


class REINFORCEAgent(Agent):
    """
    REINFORCE algorithm with policy gradients.
    Implements the policy gradient theorem: ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * G_t]
    """
    def __init__(self, env: Environment, policy_network, optimizer: SGD, discount_factor=0.99):
        self.env = env
        self.policy = StochasticPolicy(policy_network)
        self.optimizer = optimizer
        self.discount_factor = discount_factor

        # For storing episode trajectories
        self.episode_log_probs = []  # Will store tensor objects
        self.episode_rewards = []

    def act(self, state):
        action = self.policy.select_action(state)

        # Also compute and store the log probability tensor for this action
        # This is needed for the policy gradient computation
        if not isinstance(state, Tensor):
            state = Tensor(state)
        if state.data.ndim == 1:
            state.data = state.data.reshape(1, -1)

        # Get logits from network
        logits = self.policy.network(state)

        # Convert to log probabilities
        log_probs = logits - logits.exp().sum(axis=-1, keepdims=True).log()

        # Store the log probability of the chosen action (as a tensor)
        self.current_log_prob = log_probs[0, action]

        return action

    def learn(self, state, action, reward, next_state, done):
        """
        Store experience for later learning (REINFORCE is episodic).
        """
        self.episode_log_probs.append(self.current_log_prob)
        self.episode_rewards.append(reward)

        # If episode is done, perform the policy gradient update
        if done:
            self._update_policy()
            # Reset for next episode
            self.episode_log_probs = []
            self.episode_rewards = []

    def _update_policy(self):
        """
        Perform the policy gradient update using the stored trajectory.
        """
        if not self.episode_rewards:
            return

        # Compute discounted returns (G_t for each timestep)
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.discount_factor * G
            returns.insert(0, G)

        # Normalize returns (optional, but often helpful)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy gradient loss
        # Loss = -sum(log_prob * return) for all timesteps
        policy_loss = Tensor(np.array(0.0))

        for log_prob_tensor, G in zip(self.episode_log_probs, returns):
            # log_prob_tensor is already a Tensor connected to the computation graph
            # G is a scalar return
            G_tensor = Tensor(np.array(G))

            # Loss contribution: -log_prob * G (negative because we want to maximize returns)
            loss_contribution = -log_prob_tensor * G_tensor
            policy_loss = policy_loss + loss_contribution

        # Backpropagate and update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
