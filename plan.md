# RLNexus: A Comprehensive Reinforcement Learning Library

This document outlines the development plan for `RLNexus`, a Python-based Reinforcement Learning library built from scratch with minimal dependencies.

_Last reviewed: 2026-10-27._

## 1. Vision and Goals

- **Educational Focus:** Build RL algorithms from the ground up to foster a deep understanding of their inner workings.
- **Minimalism:** Rely only on a core numerical backend (`NumPy`/`CuPy`) to reduce complexity and external dependencies.
- **Performance:** Support both CPU and GPU computation to enable efficient and scalable experiments.
- **Self-Sufficiency:** Include a suite of simple, customizable environments for testing and validation, removing the need for external libraries like OpenAI Gym.
- **Proven, not just written:** An algorithm counts as done only when a benchmark test shows it learns its target environment (see §7). Unit tests that check shapes and plumbing are necessary but not sufficient.

## 2. Tech Stack

- **Core Language:** Python 3.10+ (matches `pyproject.toml`)
- **Numerical Backend:**
    - `NumPy` (2.x): Default for CPU computations.
    - `CuPy`: Optional dependency for GPU acceleration.
- **Dependency Management:** Poetry
- **Testing:** `pytest` (fast unit tests by default; slow learning benchmarks behind a `@pytest.mark.benchmark` marker)
- **Code Quality:** `black` for formatting, `ruff` for linting (not yet configured).
- **Optional extras (never required by the core):** `matplotlib` for learning-curve plots, a `gymnasium` adapter for cross-checking agents against reference environments.

## 3. Project Structure

```
/
├── rlnexus/
│   ├── algorithms/       # RL algorithms (Q-learning, DQN, REINFORCE, Actor-Critic, ...)
│   ├── core/             # Core abstractions (Agent, Environment, Policy, ReplayBuffer, spaces)
│   ├── environments/     # Custom, simple environments
│   ├── linalg/           # Tensor, autograd, NN layers, optimizers
│   ├── planning/         # Dynamic programming (policy/value iteration)
│   ├── search/           # Classical search (BFS, DFS, A*, Dijkstra) and later MCTS
│   └── utils/            # Seeding, logging, backend switcher, wrappers
├── benchmarks/           # (planned) Learning benchmarks + recorded results
├── tests/                # Unit and integration tests
├── examples/             # Practical usage examples and notebooks
├── docs/                 # MkDocs documentation
├── plan.md               # This file
├── pyproject.toml
└── README.md
```

## 4. Foundational Concepts (To be implemented/documented)

### Part 1: Mathematical & Conceptual Foundations
- **Core RL Concepts:** Agent-Environment loop, States, Actions, Rewards, Policies (π), Value Functions (V-function, Q-function), Models.
- **Mathematical Prerequisites:**
    - **Probability & Statistics:** Random variables, expectation, variance, importance sampling.
    - **Linear Algebra:** Vectors, matrices, tensor operations.
    - **Calculus & Optimization:** Gradients, chain rule, gradient descent.
- **Markov Decision Processes (MDPs):**
    - The Markov Property.
    - Transition Probabilities and Reward Functions.
    - **The Bellman Equations:** Bellman Expectation and Optimality Equations.

### Part 2: Custom Environments for Testing
- **Grid World:** A 2D grid for navigating to a goal. Ideal for visualizing and testing tabular methods. ✅
- **CartPole:** Balancing a pole on a cart; continuous state, discrete actions. ✅
- **Multi-Armed Bandits:** The classic exploration vs. exploitation problem. ⏳
- **Mountain Car:** An underpowered car must build momentum to climb a hill. ⏳

## 5. Current Status

### ✅ Built (Phases 1–2)

**Phase 1: Mathematical Foundations & Autograd Engine**
- `Tensor` with reverse-mode autograd: `+ - * / **`, `dot`, `transpose`, indexing, `relu`, `exp`, `log`, `softmax`, `sum`, `mean`
- Neural network components: `Linear`, `ReLU`, `Sequential`, `SGD`

**Phase 2: Core RL Framework & Algorithms**
- **Core Abstractions**: `Environment`, `Agent`, `Policy`, `ModelBasedEnvironment`, `ReplayBuffer`
- **Environments**: `GridWorld` (model-based, deterministic), `CartPole`
- **Tabular Methods**: Q-Learning, SARSA with `EpsilonGreedyPolicy`
- **Deep RL**: DQN with Experience Replay
- **Planning & DP**: Policy evaluation, Policy/Value Iteration, policy extraction
- **Policy Gradients**: REINFORCE, Actor-Critic with entropy bonus
- **Search Algorithms**: BFS, DFS, A*, Dijkstra with heuristics
- **Testing**: 61 tests passing
- **Documentation**: MkDocs site with theory pages and API reference

### ⚠️ Known Problems (found in the 2026-10-27 review)

**The deep RL agents do not learn CartPole.** Measured average return vs. a uniform random policy (21.4):

| Agent        | Avg return | vs random |
|--------------|-----------:|-----------|
| REINFORCE    | ~8.6       | worse     |
| DQN          | ~9.2       | worse     |
| Actor-Critic | ~10        | worse     |

The tabular methods (Q-Learning, SARSA) and DP planning do work on GridWorld.

**Autograd bugs** (`rlnexus/linalg/core.py`):
- `sum(axis=k)` backward crashes when `keepdims=False` and the reduced axis isn't the last one.
- `mean(axis=k)` divides by the total element count, not the size of axis `k`, so values are wrong.
- `dot` backward crashes for matrix·vector.
- No `__rsub__` / `__rtruediv__`, so `2 - t` and `1 / t` raise `TypeError`.
- No `detach()` / `no_grad`, so targets can leak gradients (Actor-Critic's critic targets do).

**Algorithm issues:**
- DQN has no target network, uses MSE rather than Huber, and has a fixed ε.
- Actor-Critic bootstraps from non-detached `V(s')`.
- Only plain SGD is available; there is no gradient clipping.

**API inconsistencies:**
- GridWorld returns tuple states but CartPole returns `Tensor`s.
- `SARSAAgent.learn` has a different signature from `Agent.learn`.
- `dijkstra_search` isn't exported from `rlnexus.search`.
- There is no seeding utility, and no `truncated` flag or time limit (CartPole has no episode cap).

**Test gaps:** there are no tests for Q-Learning, SARSA, DQN, or either environment, and none check that any agent actually learns.

**Tooling:**
- `poetry` is broken on the dev machine.
- `pyproject.toml` pins `numpy~=1.26`, but development runs on NumPy 2.x / Python 3.14.
- Examples only run with `PYTHONPATH=.`.
- `mkdocs.yml` has a placeholder `repo_url`.
- The README is a two-line stub.

**Git:** the `feature/actor-critic` commit with Actor-Critic and Search (`4e257ff`) is pushed but not merged to `main`. PR #3 only merged the REINFORCE commit.

---

## 6. 🚀 Development Roadmap

Ordering principle: every phase only builds on things that are already proven to learn. Each phase lists its **prerequisites** (autograd ops, environments, infrastructure) so nothing gets started on a shaky base.

### **Phase 2.5: Hardening** 🎯 *NEXT PRIORITY*

**Goal**: Make the existing library correct and trustworthy before adding algorithms.

#### 2.5a Autograd correctness
- Fix `sum`/`mean` with `axis`, matrix·vector `dot`, and add `__rsub__`, `__rtruediv__`.
- Add `detach()` and a `no_grad()` context manager.
- **Gradient-check test suite**: compare every op's backward pass against central finite differences in float64.
- New ops needed by later phases:
  - `sigmoid`, `tanh`, `abs`, `sqrt`, `clip`, `maximum`/`minimum`, `where`
  - `reshape`, `concat`/`stack`
  - `max` (with gradient routing), `logsumexp`, `log_softmax`
- Losses: `mse`, `huber`, `cross_entropy`.

#### 2.5b NN toolkit
- Optimizers: `Adam`, `RMSProp`, SGD with momentum.
- Global-norm gradient clipping.
- Recursive `Module.parameters()`, `state_dict()` / `load_state_dict()` (needed for target networks), save/load to `.npz`.
- Init schemes: Xavier, He, orthogonal.

#### 2.5c Environment & agent API
- `Discrete` / `Box` spaces with `sample()` and `contains()`.
- Consistent `step()` returning `(obs, reward, terminated, truncated, info)`. Truncation matters for correct bootstrapping.
- `TimeLimit` wrapper (CartPole capped at 500), `RecordEpisodeStatistics` wrapper.
- Observations as plain `np.ndarray` everywhere; agents convert to `Tensor` internally.
- One `Agent.learn()` signature (SARSA receives the next action via a transition object).
- Global `seed()` utility so examples and tests are reproducible.

#### 2.5d Make the existing agents learn
- **DQN**: target network (hard/soft update), Huber loss, ε-decay schedule, Adam → **Double DQN** as a flag.
- **REINFORCE**: Adam, optional learned value baseline (REINFORCE-with-baseline).
- **Actor-Critic**: detached TD targets, one-step online variant plus the current episodic variant.

#### 2.5e Testing, packaging, CI
- Unit tests for Q-Learning, SARSA, DQN, GridWorld, CartPole.
- Benchmark tests (marked slow) asserting each agent clears its threshold in §7.
- Fix packaging so `pip install -e .` works and examples run without `PYTHONPATH`; relax the NumPy pin to `>=1.26`.
- GitHub Actions running `ruff` + fast tests on every push and benchmarks nightly.
- Real README with a quick-start, `mkdocs.yml` `repo_url`, and a recorded benchmark results table in the docs.

---

### **Phase 3: Complete the Tabular Toolkit** *(Sutton & Barto, Part I)*

**Goal**: Finish the classic methods the original plan promised. They are cheap to build, high in teaching value, and give ground-truth checks for later phases.

**Environments**:
- `MultiArmedBandit` (Gaussian and Bernoulli arms, plus a non-stationary variant).
- `FrozenLake`-style **stochastic** GridWorld. DP is currently only tested on deterministic transitions.
- `CliffWalking` (the classic SARSA vs Q-learning contrast), `WindyGridWorld`, `Blackjack` (the classic Monte Carlo example).
- `MountainCar` with discrete actions.

**Algorithms**:
- **Bandits**: ε-greedy, optimistic initial values, UCB1, gradient bandit, Thompson sampling (moved up from Phase 5a).
- **Monte Carlo**: first-visit/every-visit prediction, MC control with exploring starts, off-policy MC with (weighted) importance sampling.
- **TD family**: Expected SARSA, Double Q-Learning, n-step TD/SARSA, TD(λ) and SARSA(λ) with eligibility traces.
- **Planning + learning**: Dyna-Q, Dyna-Q+, prioritized sweeping (moved up from Phase 6a).
- **Linear function approximation**: tile coding, semi-gradient SARSA on MountainCar, LSTD.

---

### **Phase 4: Deep Value-Based Methods** *(towards Rainbow)*

**Prerequisites**: Phase 2.5.

- Dueling DQN.
- Prioritized Experience Replay (sum-tree).
- n-step returns.
- Noisy Nets for exploration.
- **C51** (moved from 7a) and **QR-DQN**, which are natural DQN extensions.
- **Rainbow-lite**: combine the above with per-component ablation flags. This is a great teaching artifact.

---

### **Phase 5: Continuous Control** *(was Phase 3)*

**Prerequisites**: `tanh`, Gaussian log-prob, `Box` spaces, soft target updates (`state_dict`).

**Environments**:
- `Pendulum` (swing-up).
- Continuous-force `CartPole`.
- `MountainCarContinuous`.
- A simple 2D point-mass reacher.

**Algorithms**:
- `GaussianPolicy` (stochastic) and `DeterministicPolicy` μ_θ(s).
- **DDPG**: target networks with soft updates (τ ≈ 0.005), Ornstein-Uhlenbeck and Gaussian exploration noise.
- **TD3**: twin critics, delayed actor updates, target policy smoothing.
- **SAC**: tanh-squashed Gaussian, automatic entropy tuning.
- *(Stretch)* D4PG-style distributional critic, reusing C51 from Phase 4.

---

### **Phase 6: Modern Policy Optimization** *(was Phase 4a/4b)*

**Prerequisites**: vectorized environments (`SyncVectorEnv`, then a multiprocessing variant), Phase 5's Gaussian policy for the continuous variants.

- **GAE** (Generalized Advantage Estimation).
- **A2C**: synchronous, vectorized environments.
- **PPO**: clipped objective, value clipping, minibatch epochs; both discrete and continuous. Treat this as the flagship algorithm.
- **TRPO**: conjugate gradient plus Fisher-vector products. This needs Hessian-vector products, either by finite differences or by double-backward support in autograd. Decide which in Phase 2.5a.
- *(Stretch)* **A3C** / **IMPALA** with V-trace. Python's GIL makes thread-based A3C a poor fit, so do multiprocess rollout workers with a central learner instead.

---

### **Phase 7: Exploration** *(was Phase 5a)*

- **Contextual bandits**: LinUCB, linear Thompson sampling, neural bandits.
- **Count-based**: tabular bonuses, hash-based pseudo-counts.
- **Curiosity**: ICM and **RND** (Random Network Distillation, which is simpler and often stronger than ICM).
- **Hard-exploration environments**: DeepSea, and a key-door GridWorld with sparse reward.

---

### **Phase 8: Model-Based RL & Search** *(was Phase 6)*

**Prerequisites**: turn-based game environments (TicTacToe, Connect-4); `rlnexus.search` already gives a home for MCTS.

- **MCTS / UCT**: generic over any environment that supports cloning state.
- **AlphaZero-lite**: policy-value network with self-play on TicTacToe/Connect-4.
- **Learned dynamics**: MLP dynamics model, MPC with random shooting and CEM, **PETS** (probabilistic ensembles).
- **World Models**: VAE + latent dynamics, or a small Dreamer-style model. This needs a sampling (reparameterization) op.
- *(Stretch)* **MuZero** on TicTacToe.

---

### **Phase 9: Imitation, Inverse & Offline RL** *(NEW)*

Motivation: the library can already produce experts (DP, trained agents), so it can generate its own demonstration and offline datasets.

- **Imitation**: Behavioral Cloning, DAgger, GAIL.
- **Inverse RL**: MaxEnt IRL on GridWorld. This is an excellent fit because GridWorld exposes its model.
- **Offline RL**: a dataset format plus generators (expert, medium, random), BCQ, CQL, IQL.
- *(After Phase 11)* Decision Transformer.

---

### **Phase 10: Multi-Agent RL** *(was Phase 5b)*

**Prerequisites**: a multi-agent environment API and environments:
- matrix games (Prisoner's Dilemma, Matching Pennies);
- a predator-prey GridWorld;
- TicTacToe self-play, reusing Phase 8's environment.

**Algorithms**:
- Independent Q-learning.
- Minimax-Q and fictitious play for zero-sum games.
- **MADDPG**, **VDN** → **QMIX**, **COMA**.

---

### **Phase 11: Partial Observability & Memory** *(was Phase 8)*

**Prerequisites**: recurrent layers in autograd, unrolled through time (BPTT); `LayerNorm`; batched matmul; attention.

- **POMDP environments**: Tiger, a GridWorld with limited view, a memory/T-maze task.
- **Belief states**: exact belief update and POMDP value iteration for Tiger, then a particle filter.
- **Recurrent agents**: `RNN` / `GRU` / `LSTM` layers, DRQN, recurrent PPO.
- **Attention**: a small Transformer encoder for agents (unlocks Decision Transformer in Phase 9).
- *(Dropped)* Neural Turing Machines: high effort, low teaching value relative to the rest of the plan.

---

### **Phase 12: Advanced Topics** *(was 4c + 7b, plus new)*

- **Risk-sensitive & safe RL**: CVaR objectives, Constrained MDPs, Lagrangian PPO, CPO.
- **Hierarchical RL**: Options framework (intra-option Q-learning), Option-Critic.
- **Meta-RL**: MAML (needs second-order gradients or the first-order variant), Reptile, RL² (needs Phase 11's recurrent layers).
- **RL from Human Feedback (toy scale)**: a reward model from pairwise preferences (Bradley-Terry), PPO against the learned reward, DPO on a small sequence task. This connects the library to how LLMs are post-trained.

---

### **Cross-Cutting Infrastructure** (pull in as phases need them)

- **Backend switcher**: NumPy ↔ CuPy, with a `float32`/`float64` policy.
- **Logging**: CSV/JSONL episode logs; optional `matplotlib` learning-curve plotting.
- **Evaluation protocol**: multiple seeds, mean ± CI reporting, separate eval episodes with greedy policies.
- **Hyperparameter configs**: one dataclass per algorithm with defaults tuned to its benchmark.
- **Gymnasium adapter** (optional extra): run RLNexus agents on reference environments to validate against known results.
- **Profiling**: keep the autograd graph overhead in check (benchmark training steps/sec per algorithm).

---

## 7. 🎯 Benchmark Targets (definition of "done")

An algorithm ships only when its benchmark passes on at least 3 seeds:

| Algorithm(s)                 | Environment                 | Target                                       |
|------------------------------|-----------------------------|----------------------------------------------|
| Policy/Value Iteration       | GridWorld, stochastic grid  | Matches the analytic optimal policy          |
| Q-Learning, SARSA, MC, TD(λ) | GridWorld / CliffWalking    | Greedy policy reaches the goal on the optimal path |
| Bandit algorithms            | 10-armed testbed            | Reproduces the S&B curves (UCB > ε-greedy)   |
| Semi-gradient SARSA          | MountainCar                 | Solves in < 200 steps on average             |
| REINFORCE / A2C / PPO        | CartPole (500 cap)          | Average return ≥ 195 over 100 episodes       |
| DQN family                   | CartPole (500 cap)          | Average return ≥ 195 over 100 episodes       |
| DDPG / TD3 / SAC             | Pendulum                    | Average return ≥ −200                        |
| AlphaZero-lite               | TicTacToe                   | Never loses to a perfect minimax player      |

---

## 8. Immediate Next Steps

1. Commit this plan and merge `feature/actor-critic` (Actor-Critic + Search) into `main`.
2. **Phase 2.5a**: autograd fixes plus the gradient-check test suite.
3. **Phase 2.5b**: Adam, `state_dict`, `detach`.
4. **Phase 2.5d**: DQN, REINFORCE and Actor-Critic each beat CartPole ≥ 195, enforced by benchmark tests.
5. **Phase 2.5c/e**: API consistency, packaging, CI, README.
6. Then **Phase 3** (bandits + Monte Carlo + TD(λ)) as a fast, low-risk win before continuous control.

This roadmap keeps RLNexus educational and minimal-dependency while making sure every algorithm in it is demonstrably correct, not just implemented.
