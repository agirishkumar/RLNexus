from rlnexus.environments.cartpole import CartPole
from rlnexus.algorithms.dqn import DQNAgent
from rlnexus.linalg.nn import Sequential, Linear, ReLU, SGD
from rlnexus.linalg.core import Tensor

def main():
    # Environment
    env = CartPole()
    
    # Network
    in_features = env.observation_space_shape[0]
    num_actions = len(env.action_space)
    
    model = Sequential(
        Linear(in_features, 64),
        ReLU(),
        Linear(64, 64),
        ReLU(),
        Linear(64, num_actions)
    )
    
    # Optimizer
    optimizer = SGD(model.parameters(), lr=0.001)
    
    # Agent
    agent = DQNAgent(env, model, optimizer, epsilon=0.1, buffer_size=10000, batch_size=64)
    
    # Training loop
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
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last 10): {avg_reward:.2f}")

    print("\nTraining finished.")

if __name__ == "__main__":
    main()
