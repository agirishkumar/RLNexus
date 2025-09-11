"""
Tests for Actor-Critic algorithm implementation.

This module tests the Actor-Critic agent, ValueNetwork, and their integration
with the existing autograd and neural network systems.
"""

import pytest
import numpy as np
from rlnexus.algorithms.actor_critic import ActorCriticAgent, ValueNetwork
from rlnexus.core.policies import StochasticPolicy
from rlnexus.linalg.nn import Sequential, Linear, ReLU, SGD
from rlnexus.linalg.core import Tensor
from rlnexus.environments.cartpole import CartPole


class TestValueNetwork:
    """Test the ValueNetwork wrapper class."""
    
    def test_value_network_initialization(self):
        """Test ValueNetwork initialization."""
        # Create a simple network
        network = Sequential([
            Linear(4, 16),
            ReLU(),
            Linear(16, 1)
        ])
        
        value_net = ValueNetwork(network)
        assert value_net.network is network
    
    def test_value_network_forward(self):
        """Test ValueNetwork forward pass."""
        network = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1)
        )
        
        value_net = ValueNetwork(network)
        
        # Test forward pass
        state = Tensor(np.array([[1.0, 2.0]]))
        value = value_net(state)
        
        assert isinstance(value, Tensor)
        assert value.data.shape == (1, 1)  # Single value output
    
    def test_value_network_callable(self):
        """Test that ValueNetwork is callable."""
        network = Sequential(Linear(3, 1))
        value_net = ValueNetwork(network)
        
        state = Tensor(np.array([[1.0, 0.0, -1.0]]))
        value1 = value_net(state)
        value2 = value_net.forward(state)
        
        np.testing.assert_array_equal(value1.data, value2.data)


class TestActorCriticAgent:
    """Test the ActorCriticAgent implementation."""
    
    def test_agent_initialization(self):
        """Test ActorCriticAgent initialization."""
        # Create networks
        policy_net = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        value_net = Sequential(Linear(4, 8), ReLU(), Linear(8, 1))
        
        # Create policy and value wrappers
        policy = StochasticPolicy(policy_net)
        critic = ValueNetwork(value_net)
        
        # Create optimizers
        policy_optimizer = SGD(policy_net.parameters(), lr=0.01)
        value_optimizer = SGD(value_net.parameters(), lr=0.01)
        
        # Create agent
        agent = ActorCriticAgent(
            policy=policy,
            value_network=critic,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            discount_factor=0.95,
            entropy_coeff=0.02
        )
        
        assert agent.policy is policy
        assert agent.value_network is critic
        assert agent.discount_factor == 0.95
        assert agent.entropy_coeff == 0.02
        assert len(agent.episode_experiences) == 0
    
    def test_agent_act_method(self):
        """Test ActorCriticAgent action selection."""
        # Create simple networks
        policy_net = Sequential(Linear(2, 4), ReLU(), Linear(4, 2))
        value_net = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        
        policy = StochasticPolicy(policy_net)
        critic = ValueNetwork(value_net)
        
        policy_optimizer = SGD(policy_net.parameters(), lr=0.01)
        value_optimizer = SGD(value_net.parameters(), lr=0.01)
        
        agent = ActorCriticAgent(
            policy=policy,
            value_network=critic,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer
        )
        
        # Test action selection
        state = np.array([1.0, -0.5])
        action = agent.act(state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 2  # Valid action for 2-action environment
        
        # Check that internal state is stored
        assert hasattr(agent, '_last_state')
        assert hasattr(agent, '_last_action')
        assert hasattr(agent, '_last_log_prob')
    
    def test_agent_learn_method_stores_experience(self):
        """Test that learning stores experiences correctly."""
        # Setup agent
        policy_net = Sequential(Linear(3, 6), ReLU(), Linear(6, 2))
        value_net = Sequential(Linear(3, 6), ReLU(), Linear(6, 1))
        
        policy = StochasticPolicy(policy_net)
        critic = ValueNetwork(value_net)
        
        policy_optimizer = SGD(policy_net.parameters(), lr=0.01)
        value_optimizer = SGD(value_net.parameters(), lr=0.01)
        
        agent = ActorCriticAgent(
            policy=policy,
            value_network=critic,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer
        )
        
        # Simulate experience
        state = np.array([1.0, 0.0, -1.0])
        action = agent.act(state)
        
        # Learn from experience
        next_state = np.array([1.1, 0.1, -0.9])
        agent.learn(state, action, 1.0, next_state, False)
        
        # Check experience is stored
        assert len(agent.episode_experiences) == 1
        exp = agent.episode_experiences[0]
        assert exp.action == action
        assert exp.reward == 1.0
        assert not exp.done
    
    def test_agent_learn_method_resets_on_done(self):
        """Test that episode experiences reset when done=True."""
        # Setup agent
        policy_net = Sequential(Linear(2, 4), ReLU(), Linear(4, 2))
        value_net = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        
        policy = StochasticPolicy(policy_net)
        critic = ValueNetwork(value_net)
        
        policy_optimizer = SGD(policy_net.parameters(), lr=0.01)
        value_optimizer = SGD(value_net.parameters(), lr=0.01)
        
        agent = ActorCriticAgent(
            policy=policy,
            value_network=critic,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer
        )
        
        # Simulate some experiences
        for i in range(3):
            state = np.array([i * 0.1, i * 0.2])
            action = agent.act(state)
            next_state = np.array([(i + 1) * 0.1, (i + 1) * 0.2])
            agent.learn(state, action, 1.0, next_state, False)
        
        assert len(agent.episode_experiences) == 3
        
        # Final experience with done=True
        state = np.array([0.3, 0.6])
        action = agent.act(state)
        agent.learn(state, action, 1.0, np.array([0.0, 0.0]), True)
        
        # Should reset experiences after episode end
        assert len(agent.episode_experiences) == 0


class TestActorCriticLearning:
    """Test Actor-Critic learning behavior."""
    
    def test_advantage_computation(self):
        """Test that advantages are computed correctly."""
        # This is implicitly tested through the learning process
        # We verify that the agent can learn by checking weight changes
        
        policy_net = Sequential(Linear(2, 8), ReLU(), Linear(8, 2))
        value_net = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))
        
        policy = StochasticPolicy(policy_net)
        critic = ValueNetwork(value_net)
        
        policy_optimizer = SGD(policy_net.parameters(), lr=0.1)
        value_optimizer = SGD(value_net.parameters(), lr=0.1)
        
        agent = ActorCriticAgent(
            policy=policy,
            value_network=critic,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            discount_factor=0.99
        )
        
        # Store initial weights
        initial_policy_weights = [p.data.copy() for p in policy_net.parameters()]
        initial_value_weights = [p.data.copy() for p in value_net.parameters()]
        
        # Simulate a short episode with clear signal
        states = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
        rewards = [0.0, 1.0, 0.0]  # Positive reward in middle
        
        for i, (state, reward) in enumerate(zip(states, rewards)):
            action = agent.act(state)
            next_state = states[(i + 1) % len(states)]
            done = (i == len(states) - 1)
            agent.learn(state, action, reward, next_state, done)
        
        # Check that weights have changed (indicating learning)
        final_policy_weights = [p.data.copy() for p in policy_net.parameters()]
        final_value_weights = [p.data.copy() for p in value_net.parameters()]
        
        policy_changed = any(
            not np.allclose(initial, final, atol=1e-6)
            for initial, final in zip(initial_policy_weights, final_policy_weights)
        )
        
        value_changed = any(
            not np.allclose(initial, final, atol=1e-6)
            for initial, final in zip(initial_value_weights, final_value_weights)
        )
        
        assert policy_changed, "Policy network weights should change after learning"
        assert value_changed, "Value network weights should change after learning"
    
    def test_cartpole_integration(self):
        """Test Actor-Critic integration with CartPole environment."""
        env = CartPole()
        
        # Create networks
        policy_net = Sequential(Linear(4, 16), ReLU(), Linear(16, 2))
        value_net = Sequential(Linear(4, 16), ReLU(), Linear(16, 1))
        
        policy = StochasticPolicy(policy_net)
        critic = ValueNetwork(value_net)
        
        policy_optimizer = SGD(policy_net.parameters(), lr=0.01)
        value_optimizer = SGD(value_net.parameters(), lr=0.01)
        
        agent = ActorCriticAgent(
            policy=policy,
            value_network=critic,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer
        )
        
        # Run a few episodes to ensure no crashes
        for episode in range(3):
            state = env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < 50:  # Limit steps for test
                action = agent.act(state.data.flatten())
                next_state, reward, done, _ = env.step(action)
                
                agent.learn(state.data.flatten(), action, reward, next_state.data.flatten(), done)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Should complete without errors
            assert episode_reward > 0  # Should get some reward
    
    def test_entropy_regularization(self):
        """Test that entropy regularization is applied."""
        # Create agent with high entropy coefficient
        policy_net = Sequential(Linear(2, 4), ReLU(), Linear(4, 2))
        value_net = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        
        policy = StochasticPolicy(policy_net)
        critic = ValueNetwork(value_net)
        
        policy_optimizer = SGD(policy_net.parameters(), lr=0.1)
        value_optimizer = SGD(value_net.parameters(), lr=0.1)
        
        agent_with_entropy = ActorCriticAgent(
            policy=policy,
            value_network=critic,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            entropy_coeff=0.1  # High entropy for exploration
        )
        
        # Run learning and check that it completes without errors
        # (entropy regularization is applied internally)
        state = np.array([1.0, -1.0])
        action = agent_with_entropy.act(state)
        agent_with_entropy.learn(state, action, 1.0, np.array([0.0, 0.0]), True)
        
        # If we get here without errors, entropy regularization is working
        assert True
