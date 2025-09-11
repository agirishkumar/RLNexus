# Planning & Dynamic Programming

This section covers **planning** algorithms that work with model-based environments. These algorithms use the environment's transition model to compute optimal policies without needing to interact with the environment through trial and error.

## What is Planning?

Planning algorithms assume access to a **model** of the environment - a description of the transition probabilities `P(s'|s,a)` and rewards `R(s,a,s')`. This allows them to compute optimal policies by solving the Bellman equations analytically.

## Bellman Equations

The foundation of all planning algorithms are the **Bellman equations**:

### Bellman Expectation Equation
For a policy π, the value function satisfies:
```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V^π(s')]
```

### Bellman Optimality Equation
For the optimal policy π*, the value function satisfies:
```
V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V*(s')]
```

## Core Algorithms

### Policy Evaluation

Policy evaluation computes the value function for a given policy:

```python
from rlnexus.environments.grid_world import GridWorld
from rlnexus.planning.dp import policy_evaluation

env = GridWorld(size=4)
policy = {}  # Define your policy here
V = policy_evaluation(policy, env.model, discount_factor=0.99)
```

### Value Iteration

Value iteration directly finds the optimal value function:

```python
from rlnexus.planning.dp import value_iteration, extract_policy_from_value_function

# Find optimal value function
V_optimal = value_iteration(env)

# Extract optimal policy
policy_optimal = extract_policy_from_value_function(V_optimal, env)
```

### Policy Iteration

Policy iteration alternates between policy evaluation and policy improvement:

```python
from rlnexus.planning.dp import policy_iteration

policy_optimal, V_optimal = policy_iteration(env)
```

## Model-Based Environments

Planning algorithms require environments that expose their transition model. RLNexus provides the `ModelBasedEnvironment` base class:

```python
from rlnexus.core.mdp import ModelBasedEnvironment

class MyEnvironment(ModelBasedEnvironment):
    @property
    def model(self):
        # Return nested dict: model[state][action] = [(prob, next_state, reward, done), ...]
        return self._model
```

## When to Use Planning

Planning algorithms are ideal when:
- You have a complete model of the environment
- Environment interaction is expensive or dangerous
- You need guaranteed optimal policies
- The state space is small to medium size

For large state spaces or when you can't get a perfect model, learning-based approaches (like Q-Learning or DQN) are more appropriate.
