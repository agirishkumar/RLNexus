import numpy as np
from collections import defaultdict
from rlnexus.environments.grid_world import GridWorld

def manual_bellman_backup(env, state, V, discount_factor=0.99):
    """Manually compute the Bellman optimality backup for a state"""
    action_values = []
    print(f"Bellman backup for state {state}:")

    for action in env.action_space:
        action_value = 0
        print(f"  Action {action}:")

        for prob, next_state, reward, done in env.model[state][action]:
            v_next = V[next_state] if not done else 0.0
            contribution = prob * (reward + discount_factor * v_next)
            action_value += contribution
            print(f"    -> {next_state}: prob={prob}, reward={reward}, done={done}, V_next={v_next:.3f}, contrib={contribution:.3f}")

        print(f"    Total action value: {action_value:.3f}")
        action_values.append(action_value)

    best_value = max(action_values)
    best_action = np.argmax(action_values)
    print(f"Best action: {best_action} with value {best_value:.3f}")
    print(f"All action values: {action_values}")
    return best_value, best_action

def test_bellman_backup():
    env = GridWorld(size=4, rewards={'step': -0.01})

    # Use the converged values from Value Iteration
    V = defaultdict(float, {
        (0, 0): 0.9, (0, 1): 0.92, (0, 2): 0.94, (0, 3): 0.96,
        (1, 0): 0.92, (1, 1): 0.94, (1, 2): 0.96, (1, 3): 0.98,
        (2, 0): 0.94, (2, 1): 0.96, (2, 2): 0.98, (2, 3): 1.0,
        (3, 0): 0.96, (3, 1): 0.98, (3, 2): 1.0, (3, 3): 0.0
    })

    state = (2, 2)
    best_value, best_action = manual_bellman_backup(env, state, V)
    print(f"\nExpected: best action should be 3 (Right) with value ~0.966")

if __name__ == "__main__":
    test_bellman_backup()
