# RL Abstractions

RLNexus is built on a set of simple, abstract base classes that define the core components of a reinforcement learning system.

## `Environment`

The `rlnexus.core.environment.Environment` class is the interface for the worlds our agents interact with. Any custom environment must inherit from this class and implement three key methods:

- **`reset()`**: Resets the environment to a starting state and returns that state.
- **`step(action)`**: Executes an action, transitions the environment to a new state, and returns a tuple of `(next_state, reward, done, info)`.
- **`render()`**: Displays a representation of the environment's current state.

## `Agent`

The `rlnexus.core.agent.Agent` class is the interface for the learning entity. It encapsulates the learning algorithm and the policy. A custom agent must implement:

- **`act(state)`**: Takes the current state and returns an action to be taken. This is where the agent's policy is typically invoked.
- **`learn(...)`**: Takes an experience tuple (e.g., `state, action, reward, next_state, done`) and updates the agent's internal knowledge (e.g., updating the weights of a Q-network or the values in a Q-table).

## `Policy`

The `rlnexus.core.policy.Policy` class is the interface for the agent's decision-making logic. It defines the mapping from states to actions. A custom policy must implement:

- **`select_action(state)`**: Takes a state and returns an action.

This separation of concerns allows for modularity. For example, the same `EpsilonGreedyPolicy` can be used by both the `QLearningAgent` and the `SARSAAgent`, even though their learning algorithms differ.
