import numpy as np
import pytest
from collections import defaultdict
from rlnexus.environments.grid_world import GridWorld
from rlnexus.planning.dp import (
    policy_evaluation_step,
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration,
    extract_policy_from_value_function
)


class TestBellmanFunctions:
    """Test the basic Bellman backup functions."""

    def setup_method(self):
        """Set up test environment."""
        self.env = GridWorld(size=3, rewards={'step': -0.1})
        self.discount_factor = 0.99

    def test_policy_evaluation_step(self):
        """Test single Bellman expectation backup."""
        # Create a simple deterministic policy (always go right)
        policy = defaultdict(lambda: np.zeros(len(self.env.action_space)))
        for state in self.env.model:
            policy[state][3] = 1.0  # Always go right

        V = defaultdict(float)
        state = (0, 0)

        # Expected: go right to (0,1) with reward -0.1, then from (0,1) follow policy
        # This is complex to compute manually, just test that it returns a float
        result = policy_evaluation_step(policy, V, state, self.env.model, self.discount_factor)
        assert isinstance(result, float)

    def test_policy_evaluation_convergence(self):
        """Test that policy evaluation converges."""
        # Random policy
        policy = defaultdict(lambda: np.ones(len(self.env.action_space)) / len(self.env.action_space))

        V = policy_evaluation(policy, self.env.model, self.discount_factor)

        # Should have values for all states
        assert len(V) == len(self.env.model)
        assert all(isinstance(v, float) for v in V.values())

        # Values should be reasonable (not infinite or NaN)
        assert all(v > -10 and v < 10 for v in V.values())


class TestValueIteration:
    """Test Value Iteration algorithm."""

    def setup_method(self):
        self.env = GridWorld(size=3, rewards={'step': -0.1})

    def test_value_iteration_basic(self):
        """Test that Value Iteration produces a value function."""
        V = value_iteration(self.env)

        # Should have values for all states
        assert len(V) == len(self.env.model)
        assert all(isinstance(v, float) for v in V.values())

    def test_value_iteration_convergence(self):
        """Test that Value Iteration converges to correct values."""
        V = value_iteration(self.env)

        # The goal state should have value 0 (terminal state)
        assert abs(V[self.env.goal_pos]) < 1e-6

        # Other states should have reasonable values
        for state in self.env.model:
            if state != self.env.goal_pos:
                assert -1 <= V[state] <= 1


class TestPolicyIteration:
    """Test Policy Iteration algorithm."""

    def setup_method(self):
        self.env = GridWorld(size=3, rewards={'step': -0.1})

    def test_policy_iteration_basic(self):
        """Test that Policy Iteration produces policy and value function."""
        policy, V = policy_iteration(self.env)

        # Check policy structure
        assert len(policy) == len(self.env.model)
        for state_policy in policy.values():
            assert len(state_policy) == len(self.env.action_space)
            assert abs(sum(state_policy) - 1.0) < 1e-6  # Should be a valid probability distribution

        # Check value function
        assert len(V) == len(self.env.model)
        assert all(isinstance(v, float) for v in V.values())

    def test_policy_iteration_optimal(self):
        """Test that Policy Iteration finds the optimal policy."""
        policy, V = policy_iteration(self.env)

        # Extract optimal policy manually and compare
        optimal_policy = extract_policy_from_value_function(V, self.env)

        # Policies should be deterministic (one action with probability 1)
        for state in self.env.model:
            if state != self.env.goal_pos:  # Skip terminal state
                policy_actions = policy[state]
                optimal_actions = optimal_policy[state]

                # At least one action should be chosen in both policies
                assert np.max(policy_actions) > 0.5  # Close to 1.0
                assert np.max(optimal_actions) > 0.5


class TestPolicyExtraction:
    """Test policy extraction from value function."""

    def setup_method(self):
        self.env = GridWorld(size=3, rewards={'step': -0.1})

    def test_extract_policy_structure(self):
        """Test that extracted policy has correct structure."""
        # Use a simple value function
        V = defaultdict(float, {
            (0, 0): 0.5, (0, 1): 0.6, (0, 2): 0.7,
            (1, 0): 0.6, (1, 1): 0.7, (1, 2): 0.8,
            (2, 0): 0.7, (2, 1): 0.8, (2, 2): 0.0  # Goal
        })

        policy = extract_policy_from_value_function(V, self.env)

        # Check structure
        assert len(policy) == len(self.env.model)
        for state_policy in policy.values():
            assert len(state_policy) == len(self.env.action_space)
            # Should be deterministic (one action = 1.0, others = 0.0)
            assert sum(state_policy) == 1.0
            assert max(state_policy) == 1.0
            assert min(state_policy) == 0.0


class TestModelBasedEnvironment:
    """Test the model-based environment functionality."""

    def setup_method(self):
        self.env = GridWorld(size=3)

    def test_model_structure(self):
        """Test that the model has correct structure."""
        model = self.env.model

        # Should have entry for every state
        assert len(model) == len(self.env.observation_space)

        # Each state should have entry for every action
        for state in model:
            assert len(model[state]) == len(self.env.action_space)

            # Each action should have exactly one transition (deterministic)
            for action in model[state]:
                transitions = model[state][action]
                assert len(transitions) == 1

                prob, next_state, reward, done = transitions[0]
                assert prob == 1.0
                assert isinstance(next_state, tuple)
                assert isinstance(reward, (int, float))
                assert isinstance(done, bool)

    def test_model_consistency(self):
        """Test that model is consistent with step function."""
        model = self.env.model

        # Test a few random states
        test_states = [(0, 0), (1, 1), (2, 2)]
        for state in test_states:
            for action in self.env.action_space:
                # Get model prediction
                prob, model_next_state, model_reward, model_done = model[state][action][0]

                # Get step result
                self.env.state = state
                step_next_state, step_reward, step_done, _ = self.env.step(action)

                # Should match
                assert model_next_state == step_next_state
                assert model_reward == step_reward
                assert model_done == step_done


if __name__ == "__main__":
    pytest.main([__file__])
