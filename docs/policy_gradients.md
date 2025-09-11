# Policy Gradient Methods

This section covers **policy gradient** algorithms, which directly optimize the policy by computing gradients with respect to the policy parameters.

## The Policy Gradient Theorem

The foundation of policy gradient methods is the **Policy Gradient Theorem**:

```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) * G_t]
```

Where:
- `θ` are the policy parameters
- `π_θ(a|s)` is the policy (probability of action `a` in state `s`)
- `G_t` is the discounted return from timestep `t`

## REINFORCE Algorithm

REINFORCE is the simplest policy gradient algorithm. It samples complete trajectories and uses Monte Carlo returns to estimate the policy gradient.

### Key Features:
- **Episodic**: Updates policy after each complete episode
- **Monte Carlo**: Uses actual returns, not value function estimates
- **High Variance**: Can be unstable due to variance in returns

### Implementation:

```python
from rlnexus.environments.cartpole import CartPole
from rlnexus.algorithms.reinforce import REINFORCEAgent
from rlnexus.linalg.nn import Sequential, Linear, ReLU, SGD

# Create environment
env = CartPole()

# Create policy network (outputs action logits)
policy_network = Sequential(
    Linear(4, 64),  # CartPole state is 4D
    ReLU(),
    Linear(64, 2)   # 2 actions: left/right
)

# Create agent
optimizer = SGD(policy_network.parameters(), lr=0.01)
agent = REINFORCEAgent(env, policy_network, optimizer)

# Training loop
for episode in range(300):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
```

## Stochastic Policies

Policy gradient methods require **stochastic policies** that can output probability distributions over actions.

```python
from rlnexus.core.policies import StochasticPolicy
from rlnexus.linalg.nn import Sequential, Linear

# Create policy network
policy_net = Sequential(
    Linear(state_dim, hidden_dim),
    ReLU(),
    Linear(hidden_dim, action_dim)
)

# Wrap in stochastic policy (adds softmax and sampling)
policy = StochasticPolicy(policy_net)

# Sample action
action = policy.select_action(state)

# Get action probabilities
probs = policy.get_action_probabilities(state)
```

## Advantages of Policy Gradients

- **Continuous Action Spaces**: Can handle continuous actions naturally
- **Stochastic Policies**: Can represent multi-modal policies
- **Sample Efficiency**: Can learn from fewer samples than value-based methods
- **Direct Optimization**: Optimizes the quantity we actually care about (policy performance)

## Current Limitations

- **High Variance**: Policy gradients can be noisy
- **Sample Inefficient**: Often requires many samples
- **Local Optima**: Can get stuck in local optima
- **Hyperparameter Sensitive**: Learning rates and baselines are critical

## Future Extensions

The next steps will include:
- **Actor-Critic Methods**: Combining policy gradients with value functions
- **Baseline Subtraction**: Reducing variance with state-value baselines
- **Trust Region Methods**: More stable policy updates
- **Natural Gradients**: Better parameter updates using Fisher information
