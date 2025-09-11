import numpy as np
import random

from rlnexus.core.agent import Agent
from rlnexus.core.environment import Environment
from rlnexus.linalg.nn import Module, SGD
from rlnexus.core.replay_buffer import ReplayBuffer, Transition
from rlnexus.linalg.core import Tensor

class DQNAgent(Agent):
    """
    A Deep Q-Network agent with Experience Replay.
    """
    def __init__(self, env: Environment, network: Module, optimizer: SGD, 
                 discount_factor=0.99, epsilon=0.1, buffer_size=10000, batch_size=64):
        
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        self.replay_buffer = ReplayBuffer(buffer_size)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            # Create a new Tensor for network input to avoid modifying the original
            net_input_data = state.data if isinstance(state, Tensor) else state
            if net_input_data.ndim == 1:
                net_input_data = net_input_data.reshape(1, -1)
            
            q_values = self.network(Tensor(net_input_data))
            return np.argmax(q_values.data)

    def learn(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.replay_buffer.push(state.data, action, reward, next_state.data, done)

        # If buffer is not full enough, do not learn
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = Tensor(np.array(batch.state))
        action_batch = np.array(batch.action)
        reward_batch = Tensor(np.array(batch.reward))
        next_state_batch = Tensor(np.array(batch.next_state))
        done_batch = np.array(batch.done)
        
        # 1. Predict Q-values for the entire state batch
        q_pred_all = self.network(state_batch)
        # Use a gather operation to select the Q-value for the action taken
        q_pred = q_pred_all[range(self.batch_size), action_batch]

        # 2. Compute the Q-target
        q_next = self.network(next_state_batch)
        q_next_max = np.max(q_next.data, axis=1)
        
        # Compute the target Q value using numpy
        q_target_data = reward_batch.data + self.discount_factor * q_next_max * (1 - done_batch)
        q_target = Tensor(q_target_data)
        
        # 3. Compute loss (MSE)
        loss = ((q_pred - q_target)**2).mean()

        # 4. Backpropagate and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
