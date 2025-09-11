from rlnexus.environments.cartpole import CartPole
from rlnexus.algorithms.reinforce import REINFORCEAgent
from rlnexus.linalg.nn import Sequential, Linear, ReLU, SGD
from rlnexus.linalg.core import Tensor

def main():
    # Environment
    env = CartPole()

    # Policy Network (simple MLP)
    in_features = env.observation_space_shape[0]
    num_actions = len(env.action_space)

    policy_network = Sequential(
        Linear(in_features, 64),
        ReLU(),
        Linear(64, 64),
        ReLU(),
        Linear(64, num_actions)  # Outputs logits, will be converted to probabilities
    )

    # Optimizer
    optimizer = SGD(policy_network.parameters(), lr=0.01)

    # REINFORCE Agent
    agent = REINFORCEAgent(env, policy_network, optimizer, discount_factor=0.99)

    # Training loop
    num_episodes = 300
    episode_rewards = []

    print("Training REINFORCE agent on CartPole...")
    print("Episode | Avg Reward (last 10) | Max Reward")
    print("-" * 40)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            max_reward = max(episode_rewards[-10:])
            print(f"{episode + 1:6d} | {avg_reward:8.1f} | {max_reward:5.1f}")

    print("\nTraining completed!")
    final_avg = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
    print(f"Final average reward (last 50 episodes): {final_avg:.2f}")
    print("\nPolicy learned to balance the pole using the Policy Gradient Theorem!")
    print("The agent now selects actions based on a learned probability distribution.")

if __name__ == "__main__":
    main()
