"""
Actor-Critic Algorithm Implementation

This module implements the Actor-Critic method that combines policy gradients (actor)
with value function estimation (critic) to reduce variance in policy gradient estimates.

The Actor-Critic theorem states that we can replace the Monte Carlo return G_t in 
REINFORCE with the advantage A(s,a) = r + γV(s') - V(s), where V is a learned
value function. This significantly reduces variance while maintaining unbiased
gradient estimates.
"""

import numpy as np
from typing import Any, Optional, Tuple
from collections import namedtuple

from rlnexus.core.agent import Agent
from rlnexus.core.policies import StochasticPolicy
from rlnexus.linalg.core import Tensor
from rlnexus.linalg.nn import Module, SGD

# Experience tuple for storing transitions
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob'])


class ValueNetwork(Module):
    """
    Value function network (critic) that estimates V(s).
    
    This network learns to predict the expected return from any given state,
    which is used as a baseline to reduce variance in policy gradient estimates.
    """
    
    def __init__(self, network: Module):
        """
        Initialize value network.
        
        Args:
            network: Neural network that outputs a single value (state value)
        """
        super().__init__()
        self.network = network
    
    def forward(self, state: Tensor) -> Tensor:
        """Forward pass through value network."""
        return self.network(state)
    
    def __call__(self, state: Tensor) -> Tensor:
        """Make the network callable."""
        return self.forward(state)


class ActorCriticAgent(Agent):
    """
    Actor-Critic Agent implementing the advantage actor-critic algorithm.
    
    The agent maintains two networks:
    1. Actor (policy): π_θ(a|s) - selects actions
    2. Critic (value): V_φ(s) - estimates state values
    
    The actor is updated using policy gradients with advantage as the baseline:
    ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]
    
    The critic is updated using temporal difference learning:
    δ = r + γV(s') - V(s)
    ∇_φ L(φ) = E[δ * ∇_φ V_φ(s)]
    """
    
    def __init__(
        self,
        policy: StochasticPolicy,
        value_network: ValueNetwork,
        policy_optimizer: SGD,
        value_optimizer: SGD,
        discount_factor: float = 0.99,
        entropy_coeff: float = 0.01
    ):
        """
        Initialize Actor-Critic agent.
        
        Args:
            policy: Stochastic policy network (actor)
            value_network: Value function network (critic)
            policy_optimizer: Optimizer for policy network
            value_optimizer: Optimizer for value network
            discount_factor: Discount factor γ for future rewards
            entropy_coeff: Coefficient for entropy regularization
        """
        self.policy = policy
        self.value_network = value_network
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.discount_factor = discount_factor
        self.entropy_coeff = entropy_coeff
        
        # Store experiences for the current episode
        self.episode_experiences = []
    
    def act(self, state: Any) -> Any:
        """
        Select action using the policy network and store log probability.
        
        Args:
            state: Current environment state
            
        Returns:
            Selected action
        """
        # Convert state to tensor if needed
        if not isinstance(state, Tensor):
            state_tensor = Tensor(np.array(state, dtype=np.float32).reshape(1, -1))
        else:
            state_tensor = state
        
        # Select action using policy
        action = self.policy.select_action(state_tensor)
        
        # Get log probability for the selected action (needed for gradient computation)
        # We need to compute this through the network to maintain gradients
        logits = self.policy.network(state_tensor)
        action_probs = logits.softmax()
        log_prob = action_probs[0, action].log()
        
        # Store for later use in learning
        self._last_state = state_tensor
        self._last_action = action
        self._last_log_prob = log_prob
        
        return action
    
    def learn(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """
        Learn from experience using Actor-Critic updates.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        # Store experience
        experience = Experience(
            state=self._last_state,
            action=self._last_action,
            reward=reward,
            next_state=Tensor(np.array(next_state, dtype=np.float32).reshape(1, -1)) if not isinstance(next_state, Tensor) else next_state,
            done=done,
            log_prob=self._last_log_prob
        )
        self.episode_experiences.append(experience)
        
        # If episode is done, update both networks
        if done:
            self._update_networks()
            self.episode_experiences = []  # Reset for next episode
    
    def _update_networks(self) -> None:
        """
        Update both actor and critic networks using collected experiences.
        """
        if not self.episode_experiences:
            return
        
        # Compute advantages and returns for all experiences
        advantages = []
        returns = []
        
        # Work backwards through episode to compute returns and advantages
        next_value = Tensor(np.array([[0.0]]))  # Terminal state value is 0
        
        for exp in reversed(self.episode_experiences):
            if exp.done:
                # Terminal state
                current_value = self.value_network(exp.state)
                td_target = Tensor(np.array([[exp.reward]]))
                advantage = td_target - current_value
            else:
                # Non-terminal state
                current_value = self.value_network(exp.state)
                td_target = Tensor(np.array([[exp.reward]])) + self.discount_factor * next_value
                advantage = td_target - current_value
            
            advantages.insert(0, advantage)
            returns.insert(0, td_target)
            next_value = current_value
        
        # Update critic (value network)
        self._update_critic(returns)
        
        # Update actor (policy network)
        self._update_actor(advantages)
    
    def _update_critic(self, returns: list) -> None:
        """
        Update the critic network using mean squared error loss.
        
        Args:
            returns: List of TD targets for each state
        """
        total_value_loss = Tensor(np.array([[0.0]]))
        
        for i, exp in enumerate(self.episode_experiences):
            # Predict value for current state
            predicted_value = self.value_network(exp.state)
            
            # Compute value loss (MSE)
            value_error = returns[i] - predicted_value
            value_loss = value_error * value_error  # Squared error
            total_value_loss = total_value_loss + value_loss
        
        # Average loss over episode
        avg_value_loss = total_value_loss / len(self.episode_experiences)
        
        # Backpropagate value loss
        self.value_optimizer.zero_grad()
        avg_value_loss.backward()
        self.value_optimizer.step()
    
    def _update_actor(self, advantages: list) -> None:
        """
        Update the actor network using policy gradient with advantage baseline.
        
        Args:
            advantages: List of advantage estimates for each action
        """
        total_policy_loss = Tensor(np.array([[0.0]]))
        total_entropy = Tensor(np.array([[0.0]]))
        
        for i, exp in enumerate(self.episode_experiences):
            # Policy gradient loss: -log π(a|s) * A(s,a)
            advantage_detached = Tensor(advantages[i].data.copy())  # Detach from critic's graph
            policy_loss = -exp.log_prob * advantage_detached
            total_policy_loss = total_policy_loss + policy_loss
            
            # Entropy regularization (encourages exploration)
            if self.entropy_coeff > 0:
                logits = self.policy.network(exp.state)
                action_probs = logits.softmax()
                entropy = -(action_probs * action_probs.log()).sum()
                total_entropy = total_entropy + entropy
        
        # Average losses over episode
        avg_policy_loss = total_policy_loss / len(self.episode_experiences)
        avg_entropy = total_entropy / len(self.episode_experiences)
        
        # Total actor loss (policy loss - entropy bonus)
        actor_loss = avg_policy_loss - self.entropy_coeff * avg_entropy
        
        # Backpropagate policy loss
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()
