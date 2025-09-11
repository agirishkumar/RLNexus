# Getting Started

This guide will walk you through a complete example of training a Deep Q-Network (DQN) agent on the CartPole environment using RLNexus.

## 1. Setting up the Environment

First, we import and initialize our `CartPole` environment. This environment is self-contained and requires no external dependencies.

```python
from rlnexus.environments.cartpole import CartPole

env = CartPole()
```

## 2. Building the Q-Network

Next, we define a simple Multi-Layer Perceptron (MLP) to serve as our function approximator for the Q-values. We use the `Sequential`, `Linear`, and `ReLU` modules from our `linalg` library.

```python
from rlnexus.linalg.nn import Sequential, Linear, ReLU

in_features = env.observation_space_shape[0]
num_actions = len(env.action_space)

model = Sequential(
    Linear(in_features, 64),
    ReLU(),
    Linear(64, 64),
    ReLU(),
    Linear(64, num_actions)
)
```

## 3. Creating the Agent

With the environment and network defined, we can create our `DQNAgent`. We also need to provide an optimizer, like `SGD`. The agent is configured with hyperparameters for the experience replay buffer and epsilon-greedy exploration.

```python
from rlnexus.algorithms.dqn import DQNAgent
from rlnexus.linalg.nn import SGD

optimizer = SGD(model.parameters(), lr=0.001)

agent = DQNAgent(
    env=env, 
    network=model, 
    optimizer=optimizer,
    epsilon=0.1,
    buffer_size=10000,
    batch_size=64
)
```

## 4. The Training Loop

Finally, we run the training loop. The agent interacts with the environment, stores experiences in its replay buffer, and learns by sampling from the buffer.

```python
num_episodes = 500
episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    episode_rewards.append(total_reward)
    if (episode + 1) % 10 == 0:
        avg_reward = sum(episode_rewards[-10:]) / 10
        print(f"Episode {episode + 1}, Avg Reward (last 10): {avg_reward:.2f}")

print("\\nTraining finished.")
```

This complete, runnable example can be found in `examples/train_dqn_cartpole.py`.

## Alternative: Planning-Based Approaches

If you have access to a model of the environment (transition probabilities and rewards), you can use **planning algorithms** instead of reinforcement learning. These compute optimal policies analytically without needing to interact with the environment.

For example, with our `GridWorld` environment:

```python
from rlnexus.environments.grid_world import GridWorld
from rlnexus.planning.dp import value_iteration, extract_policy_from_value_function

# Create model-based environment
env = GridWorld(size=4)

# Find optimal policy using Value Iteration
V_optimal = value_iteration(env)
policy_optimal = extract_policy_from_value_function(V_optimal, env)

print("Planning found the optimal policy analytically!")
```

Planning algorithms are covered in detail in the [Planning & Dynamic Programming](planning.md) section.
