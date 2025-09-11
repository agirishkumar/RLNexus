import numpy as np
from rlnexus.environments.grid_world import GridWorld

def debug_action_values(env, state=(2, 2)):
    """Debug the action values for a specific state"""
    print(f"Debugging action values for state {state}")
    print(f"Goal is at {env.goal_pos}")

    for action in env.action_space:
        action_value = 0
        print(f"Action {action}: ", end="")

        for prob, next_state, reward, done in env.model[state][action]:
            print(f"-> {next_state} (prob={prob}, reward={reward}, done={done})", end=" ")
            # Use actual values from the value function
            v_next = env.model[next_state][0][3] if next_state in env.model else 0.94  # Done flag
            action_value += prob * (reward + 0.99 * v_next)

        print(f"= {action_value:.3f}")

if __name__ == "__main__":
    env = GridWorld(size=4, rewards={'step': -0.01})
    debug_action_values(env)
