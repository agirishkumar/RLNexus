# Welcome to RLNexus

**RLNexus** is a Python library for Reinforcement Learning, built entirely from scratch with minimal dependencies.

The primary goal of RLNexus is to provide a clear, concise, and educational implementation of the most common RL algorithms, core components, and environments. By building everything from the ground up, including the automatic differentiation engine, this library serves as a practical tool for learning the deep inner workings of RL.

## Features

- **Custom Autograd Engine**: A lightweight automatic differentiation engine built on NumPy that supports dynamic computation graphs.
- **Minimal Dependencies**: The entire library is built on top of `numpy`, avoiding heavy frameworks like PyTorch or TensorFlow.
- **Tabular & Deep RL**: Includes classic tabular methods (Q-Learning, SARSA) and modern deep RL algorithms (DQN with Experience Replay).
- **Custom Environments**: Comes with classic environments like `GridWorld` and `CartPole` built from scratch.
- **Clear Abstractions**: Designed with clean `Environment`, `Agent`, and `Policy` base classes to promote modularity and extension.
