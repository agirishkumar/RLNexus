import numpy as np
from collections import defaultdict
from rlnexus.environments.grid_world import GridWorld
from rlnexus.planning.dp import policy_evaluation

def main():
    # 1. Create the environment
    env = GridWorld(size=4)
    
    # 2. Define a random policy
    # For a random policy, each action has a probability of 1 / num_actions
    random_policy = defaultdict(lambda: np.ones(len(env.action_space)) / len(env.action_space))

    print("Evaluating random policy...")
    
    # 3. Use the policy_evaluation function
    V = policy_evaluation(random_policy, env.model)
    
    print("\nPolicy Evaluation Complete.")
    print("State-Value Function (V) for the random policy:")
    
    # 4. Print the value function grid
    grid_v = np.zeros((env.size, env.size))
    for state, value in V.items():
        grid_v[state] = value
        
    # Pretty print with 2 decimal places
    print(np.round(grid_v, 2))

if __name__ == "__main__":
    main()
