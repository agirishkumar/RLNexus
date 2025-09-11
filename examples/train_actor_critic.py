"""
Actor-Critic Training Example

This example demonstrates how to train an Actor-Critic agent on the CartPole environment.
The Actor-Critic method combines policy gradients (actor) with value function estimation (critic)
to achieve more stable and efficient learning compared to pure policy gradient methods.

Key improvements over REINFORCE:
1. Lower variance: Uses TD targets instead of Monte Carlo returns
2. Online learning: Updates after each episode instead of just at the end
3. Baseline subtraction: Critic provides baseline to reduce gradient variance
"""

import numpy as np
from rlnexus.environments.cartpole import CartPole
from rlnexus.algorithms.actor_critic import ActorCriticAgent, ValueNetwork
from rlnexus.core.policies import StochasticPolicy
from rlnexus.linalg.nn import Sequential, Linear, ReLU, SGD
from rlnexus.linalg.core import Tensor


def create_policy_network(state_dim: int, action_dim: int, hidden_dim: int = 64) -> Sequential:
    """Create a policy network for the actor."""
    return Sequential(
        Linear(state_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, action_dim)
    )


def create_value_network(state_dim: int, hidden_dim: int = 64) -> Sequential:
    """Create a value network for the critic."""
    return Sequential(
        Linear(state_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, 1)  # Single output for state value
    )


def main():
    # Environment setup
    env = CartPole()
    state_dim = 4  # [position, velocity, angle, angular_velocity]
    action_dim = 2  # [left, right]
    
    print("Training Actor-Critic Agent on CartPole")
    print("=" * 50)
    
    # Create networks
    policy_net = create_policy_network(state_dim, action_dim, hidden_dim=128)
    value_net = create_value_network(state_dim, hidden_dim=128)
    
    # Create policy and value network wrappers
    policy = StochasticPolicy(policy_net)
    critic = ValueNetwork(value_net)
    
    # Create optimizers
    policy_optimizer = SGD(policy_net.parameters(), lr=0.001)
    value_optimizer = SGD(value_net.parameters(), lr=0.005)  # Critic learns faster
    
    # Create Actor-Critic agent
    agent = ActorCriticAgent(
        policy=policy,
        value_network=critic,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        discount_factor=0.99,
        entropy_coeff=0.01  # Small entropy bonus for exploration
    )
    
    # Training parameters
    num_episodes = 500
    print_every = 50
    episode_rewards = []
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        max_steps = 500
        
        while step < max_steps:
            # Agent selects action
            action = agent.act(state)
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            
            # Agent learns from experience
            agent.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            max_reward = np.max(episode_rewards[-print_every:])
            min_reward = np.min(episode_rewards[-print_every:])
            
            print(f"Episode {episode + 1:3d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Max: {max_reward:6.1f} | "
                  f"Min: {min_reward:6.1f}")
    
    # Final performance summary
    print("\n" + "=" * 50)
    print("Training Complete!")
    
    final_avg = np.mean(episode_rewards[-100:])
    overall_avg = np.mean(episode_rewards)
    best_episode = np.max(episode_rewards)
    
    print(f"Final 100 episodes average: {final_avg:.1f}")
    print(f"Overall average reward: {overall_avg:.1f}")
    print(f"Best episode reward: {best_episode:.1f}")
    
    # Test trained agent
    print("\nTesting trained agent...")
    test_episodes = 10
    test_rewards = []
    
    for test_ep in range(test_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        
        while step < 500:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {test_ep + 1}: {episode_reward:.1f}")
    
    print(f"\nTest Average: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    
    # Compare with theoretical maximum
    print(f"\nCartPole theoretical maximum: 500.0")
    if np.mean(test_rewards) > 450:
        print("🎉 Excellent! Agent learned to balance the pole very well!")
    elif np.mean(test_rewards) > 200:
        print("✅ Good! Agent learned a reasonable policy.")
    else:
        print("⚠️  Agent still learning. Try training longer or tuning hyperparameters.")


if __name__ == "__main__":
    main()
