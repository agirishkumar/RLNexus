import numpy as np
import pytest
from rlnexus.environments.cartpole import CartPole
from rlnexus.algorithms.reinforce import REINFORCEAgent
from rlnexus.linalg.nn import Sequential, Linear, ReLU, SGD
from rlnexus.core.policies import StochasticPolicy


class TestStochasticPolicy:
    """Test the StochasticPolicy class."""

    def setup_method(self):
        """Set up test policy."""
        # Simple policy network
        self.network = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)  # 2 actions for CartPole
        )
        self.policy = StochasticPolicy(self.network)
        self.test_state = np.array([0.1, 0.2, 0.3, 0.4])

    def test_select_action(self):
        """Test action selection."""
        action = self.policy.select_action(self.test_state)
        assert action in [0, 1]  # CartPole actions

    def test_get_action_probabilities(self):
        """Test probability computation."""
        probs = self.policy.get_action_probabilities(self.test_state)
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 1e-6  # Should sum to 1
        assert all(p >= 0 for p in probs)  # All probabilities should be non-negative


class TestREINFORCEAgent:
    """Test the REINFORCE agent."""

    def setup_method(self):
        """Set up test environment and agent."""
        self.env = CartPole()

        # Simple policy network
        policy_network = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )

        optimizer = SGD(policy_network.parameters(), lr=0.01)
        self.agent = REINFORCEAgent(self.env, policy_network, optimizer)

    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.env == self.env
        assert isinstance(self.agent.policy, StochasticPolicy)
        assert self.agent.discount_factor == 0.99
        assert len(self.agent.episode_log_probs) == 0
        assert len(self.agent.episode_rewards) == 0

    def test_act_method(self):
        """Test action selection."""
        state = self.env.reset()
        action = self.agent.act(state)
        assert action in [0, 1]

        # Check that log probability was stored
        assert hasattr(self.agent, 'current_log_prob')

    def test_learn_method_stores_experience(self):
        """Test that learn method stores experience correctly."""
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = 0
        reward = 1.0
        next_state = np.array([0.2, 0.3, 0.4, 0.5])
        done = False

        # Before learning
        initial_log_probs_len = len(self.agent.episode_log_probs)
        initial_rewards_len = len(self.agent.episode_rewards)

        self.agent.act(state)  # This sets current_log_prob
        self.agent.learn(state, action, reward, next_state, done)

        # After learning (episode not done)
        assert len(self.agent.episode_log_probs) == initial_log_probs_len + 1
        assert len(self.agent.episode_rewards) == initial_rewards_len + 1

    def test_learn_method_resets_on_done(self):
        """Test that learn method resets episode data when done=True."""
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = 0
        reward = 1.0
        next_state = np.array([0.2, 0.3, 0.4, 0.5])

        # Add some experience
        self.agent.act(state)
        self.agent.learn(state, action, reward, next_state, done=False)
        assert len(self.agent.episode_log_probs) > 0
        assert len(self.agent.episode_rewards) > 0

        # Now finish episode
        self.agent.act(next_state)
        self.agent.learn(next_state, action, reward, next_state, done=True)

        # Episode data should be reset
        assert len(self.agent.episode_log_probs) == 0
        assert len(self.agent.episode_rewards) == 0


class TestREINFORCELearning:
    """Test the actual learning capability of REINFORCE."""

    def setup_method(self):
        """Set up a minimal test environment."""
        self.env = CartPole()

        # Very simple policy network for testing
        policy_network = Sequential(
            Linear(4, 4),
            Linear(4, 2)
        )

        optimizer = SGD(policy_network.parameters(), lr=0.01)  # Reasonable learning rate for testing
        self.agent = REINFORCEAgent(self.env, policy_network, optimizer, discount_factor=0.9)

    def test_policy_gradient_computation(self):
        """Test that policy gradients are computed correctly."""
        # Run a short episode
        state = self.env.reset()
        total_reward = 0
        steps = 0

        while not self.env.state[2] > 0.1 and steps < 5:  # Keep pole somewhat upright, max 5 steps
            action = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.learn(state, action, reward, next_state, done=False)
            state = next_state
            total_reward += reward
            steps += 1

        # Finish the episode
        self.agent.act(state)
        self.agent.learn(state, action, reward, next_state, done=True)

        # Check that episode data was processed
        assert len(self.agent.episode_log_probs) == 0  # Should be reset
        assert len(self.agent.episode_rewards) == 0

    def test_multiple_episodes(self):
        """Test learning over multiple short episodes."""
        initial_weights = [p.data.copy() for p in self.agent.policy.network.parameters()]

        # Run several episodes with guaranteed rewards to ensure learning occurs
        for episode in range(10):  # More episodes
            state = self.env.reset()
            done = False
            steps = 0
            episode_reward = 0

            while not done and steps < 30:  # Longer episodes for more reward
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                steps += 1

                if done:
                    break

        # Check that weights have changed (learning occurred)
        final_weights = [p.data.copy() for p in self.agent.policy.network.parameters()]

        # At least some weights should have changed
        weights_changed = False
        for initial, final in zip(initial_weights, final_weights):
            if not np.allclose(initial, final, atol=1e-7):  # More sensitive to changes
                weights_changed = True
                break

        assert weights_changed, "Policy weights should have changed after learning"


if __name__ == "__main__":
    pytest.main([__file__])
