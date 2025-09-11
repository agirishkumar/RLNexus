# Custom Environments

RLNexus includes a set of classic reinforcement learning environments, built from scratch to avoid external dependencies.

## `GridWorld`

- **File**: `rlnexus.environments.grid_world`
- **State Space**: Discrete, 2D Tuple `(row, col)`
- **Action Space**: Discrete (Up, Down, Left, Right)

A simple grid-based navigation task. The agent (`A`) must navigate from a start position to a goal (`G`), avoiding obstacles (`X`).

This environment is highly configurable. You can specify:
- `size`: The height and width of the square grid.
- `start`: The starting coordinates.
- `goal`: The goal coordinates.
- `obstacles`: A list of tuples representing obstacle coordinates.

It is ideal for testing and visualizing tabular RL algorithms like Q-Learning and SARSA.

## `CartPole`

- **File**: `rlnexus.environments.cartpole`
- **State Space**: Continuous, 4D Vector (`cart_position`, `cart_velocity`, `pole_angle`, `pole_velocity`)
- **Action Space**: Discrete (Push Left, Push Right)

The classic control problem of balancing a pole on a moving cart. The episode ends if the pole falls over too far or if the cart moves off the edge of the track. A reward of +1 is given for every step the pole remains balanced.

This environment is essential for testing algorithms that use function approximation, such as our DQN agent, as the state space is continuous and cannot be represented in a simple table.
