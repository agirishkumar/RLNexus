# RLNexus: A Comprehensive Reinforcement Learning Library

This document outlines the development plan for `RLNexus`, a Python-based Reinforcement Learning library built from scratch with minimal dependencies.

## 1. Vision and Goals

- **Educational Focus:** Build RL algorithms from the ground up to foster a deep understanding of their inner workings.
- **Minimalism:** Rely only on a core numerical backend (`NumPy`/`CuPy`) to reduce complexity and external dependencies.
- **Performance:** Support both CPU and GPU computation to enable efficient and scalable experiments.
- **Self-Sufficiency:** Include a suite of simple, customizable environments for testing and validation, removing the need for external libraries like OpenAI Gym.

## 2. Tech Stack

- **Core Language:** Python 3.9+
- **Numerical Backend:**
    - `NumPy`: Default for CPU computations.
    - `CuPy`: Optional dependency for GPU acceleration.
- **Dependency Management:** Poetry
- **Testing:** `pytest`
- **Code Quality:** `black` for formatting, `ruff` for linting.

## 3. Project Structure

```
/
├── rlnexus/
│   ├── __init__.py
│   ├── algorithms/       # RL algorithms (Q-learning, PPO, etc.)
│   ├── core/             # Core abstractions (Agent, Environment, Policy)
│   ├── environments/     # Custom, simple environments
│   ├── linalg/           # Linear algebra, autograd, and NN components
│   └── utils/            # Helper functions, backend switcher
├── tests/                # Unit and integration tests
├── examples/             # Practical usage examples and notebooks
├── docs/                 # Project documentation
├── plan.md               # This file
├── pyproject.toml
└── README.md
```

## 4. Foundational Concepts (To be implemented/documented)

### Part 1: Mathematical & Conceptual Foundations
- **Core RL Concepts:** Agent-Environment loop, States, Actions, Rewards, Policies (π), Value Functions (V-function, Q-function), Models.
- **Mathematical Prerequisites:**
    - **Probability & Statistics:** Random variables, expectation, variance.
    - **Linear Algebra:** Vectors, matrices, tensor operations.
    - **Calculus & Optimization:** Gradients, chain rule, gradient descent.
- **Markov Decision Processes (MDPs):**
    - The Markov Property.
    - Transition Probabilities and Reward Functions.
    - **The Bellman Equations:** Bellman Expectation and Optimality Equations.

### Part 2: Custom Environments for Testing
- **Grid World:** A 2D grid for navigating to a goal. Ideal for visualizing and testing tabular methods.
- **Multi-Armed Bandits:** The classic exploration vs. exploitation problem.
- **CartPole:** Balancing a pole on a cart; a classic continuous control problem.
- **Mountain Car:** An underpowered car must build momentum to climb a hill.

## 5. Revised Phased Development Roadmap

### Phase 1: Mathematical Foundations & Backend (Priority)
- **Tasks:**
    - Set up the basic project structure and testing framework (`pytest`).
    - Initially, all mathematical components will be built directly using `NumPy` to simplify development. Support for `CuPy` and a backend switcher can be added in the future.
    - In `rlnexus/linalg/`, design a core `Tensor` or `Array` class as a unified interface over `NumPy`.
    - Implement essential mathematical operations: matrix multiplication, activation functions (ReLU, Sigmoid, etc.), and loss functions (MSE), along with their gradients, to form a minimal autograd system.
    - Implement a basic `Linear` layer and a simple `Optimizer` like SGD.

### Phase 2: Core RL Abstractions & Simple Environments
- **Tasks:**
    - In `rlnexus/core/`, build the abstractions (`Environment`, `Agent`, `Policy`) using the mathematical components from Phase 1.
    - In `rlnexus/environments/`, create `GridWorld` and `MultiArmedBandit`.
    - Write tests for the core abstractions and environments.

### Phase 3: Tabular Methods
- **Tasks:**
    - In `rlnexus/algorithms/`, implement classic tabular methods:
        - Dynamic Programming (Policy/Value Iteration).
        - Monte Carlo methods.
        - Temporal-Difference methods (SARSA, Q-Learning).
    - Thoroughly test and validate these algorithms in the `GridWorld` environment.

### Phase 4: Function Approximation
- **Tasks:**
    - Assemble the mathematical primitives from Phase 1 into a simple feed-forward neural network/MLP model.
    - Integrate this model with the `Agent` and `Policy` abstractions.
    - Adapt the tabular methods to their function approximation counterparts (e.g., Deep Q-Learning).
    - Implement the continuous state-space environments (`CartPole`, `MountainCar`) to test these new capabilities.

### Phase 5: Advanced Algorithms & Finalization
- **Tasks:**
    - Implement Policy Gradient methods like REINFORCE and Actor-Critic, leveraging the autograd capabilities.
    - Benchmark the algorithms, refine the library's API, and improve performance.
    - Expand documentation in `docs/` and create detailed examples in `examples/`.
