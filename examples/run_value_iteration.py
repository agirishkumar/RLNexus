import numpy as np
from rlnexus.environments.grid_world import GridWorld
from rlnexus.planning.dp import value_iteration, extract_policy_from_value_function

def main():
    # 1. Create the environment with smaller step penalty
    env = GridWorld(size=4, rewards={'step': -0.01})
    
    print("Running Value Iteration...")
    
    # 2. Run the value iteration algorithm
    optimal_v = value_iteration(env)
    
    print("\nValue Iteration Complete.")
    print("Optimal State-Value Function (V*):")
    
    # Print the optimal value function grid
    grid_v = np.zeros((env.size, env.size))
    for state, value in optimal_v.items():
        grid_v[state] = value
    print(np.round(grid_v, 2))
    
    # 3. Extract the optimal policy
    optimal_policy = extract_policy_from_value_function(optimal_v, env)
    
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
