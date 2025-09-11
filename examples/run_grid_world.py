import random
from rlnexus.environments.grid_world import GridWorld

def main():
    # Initialize the environment
    env = GridWorld(size=5, obstacles=[(1, 1), (2, 2), (3, 3)])
    
    state = env.reset()
    done = False
    
    print("Initial State:")
    env.render()
    
    for i in range(10):
        if done:
            print("Episode finished.")
            break
            
        # Choose a random action
        action = random.choice(env.action_space)
        print(f"\nStep {i+1}: Taking action {action}")
        
        # Take a step in the environment
        state, reward, done, info = env.step(action)
        
        print(f"New State: {state}, Reward: {reward}, Done: {done}")
        env.render()

if __name__ == "__main__":
    main()
