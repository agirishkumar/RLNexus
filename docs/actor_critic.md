# Actor-Critic Methods

Actor-Critic methods represent a significant advancement in reinforcement learning by combining the strengths of value-based and policy-based approaches. This page explains the theoretical foundations and practical implementation of Actor-Critic in RLNexus.

## Theoretical Foundation

### The Actor-Critic Architecture

Actor-Critic methods maintain two components:

1. **Actor (Policy)**: π_θ(a|s) - A stochastic policy that selects actions
2. **Critic (Value Function)**: V_φ(s) - Estimates the value of states

### The Policy Gradient Theorem with Baseline

The key insight is replacing the Monte Carlo return G_t in REINFORCE with a lower-variance estimate:

```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) × A(s,a)]
```

Where the **advantage function** A(s,a) is:
```
A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)
```

### Why Actor-Critic Works Better

**REINFORCE Problems:**
- High variance due to Monte Carlo returns
- Only learns at episode end
- Requires complete episodes

**Actor-Critic Solutions:**
- Uses TD targets (lower variance)
- Learns after each step (online)
- Baseline subtraction reduces gradient variance

## Algorithm Details

### Actor Update (Policy Gradient)

The actor learns using the policy gradient with advantage baseline:

```python
# Compute advantage
advantage = reward + gamma * V(next_state) - V(state)

# Policy gradient loss
policy_loss = -log_prob * advantage.detach()  # Detach to avoid critic gradients

# Update policy
policy_loss.backward()
policy_optimizer.step()
```

### Critic Update (Value Function)

The critic learns to predict state values using TD learning:

```python
# Compute TD target
td_target = reward + gamma * V(next_state)

# Value function loss (MSE)
value_loss = (td_target - V(state))**2

# Update value function
value_loss.backward()
value_optimizer.step()
```

### Entropy Regularization

To encourage exploration, we add an entropy bonus:

```python
entropy = -(action_probs * log(action_probs)).sum()
total_loss = policy_loss - entropy_coeff * entropy
```

## Implementation Highlights

### ValueNetwork Class

```python
class ValueNetwork(Module):
    def __init__(self, network):
        self.network = network
    
    def forward(self, state):
        return self.network(state)  # Single value output
```

### ActorCriticAgent Class

```python
class ActorCriticAgent(Agent):
    def __init__(self, policy, value_network, policy_optimizer, value_optimizer):
        self.policy = policy           # StochasticPolicy
        self.value_network = value_network  # ValueNetwork
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
```

### Key Features

1. **Episodic Updates**: Collects full episodes before updating both networks
2. **Advantage Computation**: Uses TD targets to compute advantages
3. **Separate Optimizers**: Different learning rates for actor and critic
4. **Entropy Regularization**: Encourages exploration in early training

## Training Process

### Episode Collection
```python
# For each step in episode:
action = agent.act(state)
next_state, reward, done = env.step(action)
agent.learn(state, action, reward, next_state, done)
```

### Network Updates (at episode end)
```python
# Compute advantages for all steps
advantages = []
for experience in episode:
    advantage = reward + gamma * V(next_state) - V(state)
    advantages.append(advantage)

# Update critic
value_loss = mean_squared_error(td_targets, predicted_values)

# Update actor  
policy_loss = -mean(log_probs * advantages.detach())
```

## Comparison with Other Methods

| Method | Variance | Sample Efficiency | Online Learning | Stability |
|--------|----------|-------------------|-----------------|-----------|
| REINFORCE | High | Low | No | Medium |
| Q-Learning | N/A | Medium | Yes | High |
| **Actor-Critic** | **Low** | **High** | **Yes** | **High** |
| DQN | N/A | Medium | Yes | Medium |

## Advantages and Limitations

### Advantages 
- **Lower Variance**: Baseline subtraction reduces gradient variance
- **Sample Efficiency**: Learns from every transition, not just episodes
- **Online Learning**: Updates during episode, not just at end
- **General**: Works with continuous and discrete action spaces

### Limitations 
- **Complexity**: Two networks to train and tune
- **Hyperparameter Sensitivity**: Learning rates, entropy coefficient
- **Bias-Variance Tradeoff**: TD targets introduce bias but reduce variance

## Usage Example

```python
from rlnexus.algorithms.actor_critic import ActorCriticAgent, ValueNetwork
from rlnexus.core.policies import StochasticPolicy
from rlnexus.linalg.nn import Sequential, Linear, ReLU, SGD

# Create networks
policy_net = Sequential([Linear(4, 64), ReLU(), Linear(64, 2)])
value_net = Sequential([Linear(4, 64), ReLU(), Linear(64, 1)])

# Create wrappers
policy = StochasticPolicy(policy_net)
critic = ValueNetwork(value_net)

# Create optimizers
policy_optimizer = SGD(policy_net.parameters(), lr=0.001)
value_optimizer = SGD(value_net.parameters(), lr=0.005)

# Create agent
agent = ActorCriticAgent(
    policy=policy,
    value_network=critic,
    policy_optimizer=policy_optimizer,
    value_optimizer=value_optimizer,
    discount_factor=0.99,
    entropy_coeff=0.01
)

# Training loop
for episode in range(1000):
    state = env.reset()
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
```

## Next Steps

After understanding Actor-Critic, you can explore:

1. **Advanced Actor-Critic**: A3C, A2C, PPO
2. **Deterministic Policy Gradients**: DDPG, TD3, SAC  
3. **Multi-Agent**: MADDPG, COMA
4. **Hierarchical**: Option-Critic, HAC

Actor-Critic forms the foundation for many state-of-the-art RL algorithms!
