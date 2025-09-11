from rlnexus.environments.grid_world import GridWorld
from rlnexus.algorithms.sarsa import SARSAAgent

def main():
    # Environment
    env = GridWorld(size=5, obstacles=[(1, 2), (2, 2), (3, 1)])
    
    # Agent
    agent = SARSAAgent(env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1)
    
    # Training loop
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        action = agent.act(state)
        done = False
        
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = agent.act(next_state)
            
            # The learn method is slightly different for SARSA
            agent.learn(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} finished.")

    print("\nTraining finished.")
    print("Learned Q-Table (sample):")
    
    # Print some Q-values
    for i in range(5):
        for j in range(5):
            state = (i, j)
            if state in agent.q_table:
                q_vals = [f"{q:.2f}" for q in agent.q_table[state]]
                print(f"State {state}: Q-values (U,D,L,R) = {q_vals}")

if __name__ == "__main__":
    main()
