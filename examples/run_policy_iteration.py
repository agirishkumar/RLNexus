import numpy as np
from rlnexus.environments.grid_world import GridWorld
from rlnexus.planning.dp import policy_iteration

def main():
    # 1. Create the environment with a smaller step penalty for a clearer policy
    env = GridWorld(size=4, rewards={'step': -0.01})
    
    print("Running Policy Iteration...")
    
    # 2. Run the policy iteration algorithm
    optimal_policy, optimal_v = policy_iteration(env)
    
    print("\nPolicy Iteration Complete.")
    print("Optimal State-Value Function (V*):")
    
    # Print the optimal value function grid
    grid_v = np.zeros((env.size, env.size))
    for state, value in optimal_v.items():
        grid_v[state] = value
    print(np.round(grid_v, 2))
    
    print("\nOptimal Policy (π*):")
    # 0:Up, 1:Down, 2:Left, 3:Right
    action_map = ['↑', '↓', '←', '→']
    grid_policy = np.full((env.size, env.size), ' ', dtype=str)
    for state, action_probs in optimal_policy.items():
        if state == env.goal_pos:
            grid_policy[state] = 'G'
        else:
            action = np.argmax(action_probs)
            grid_policy[state] = action_map[action]
            
    print("\n".join(" ".join(row) for row in grid_policy))

if __name__ == "__main__":
    main()
