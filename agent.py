from collections import deque
import random
import numpy as np
import torch
from torch import optim
from CNN import *


class Agent:
    def __init__(self, state_size, action_size, bs, lr, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.lr = lr
        self.gamma =gamma
        self.Q_local = Q_Network(state_size, action_size)
        self.Q_target = Q_Network(state_size, action_size)
        self.initialize()
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)
        self.memory = deque(maxlen=100000)

    def action_choose(self, eps, state):
        if random.random() > eps:
            state = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_values = self.Q_local(state)
            return np.argmax(action_values.data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        experiences = random.sample(self.memory, self.bs)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()

        Q_values = self.Q_local(states)
        Q_values = torch.gather(Q_values, -1, actions)

        with torch.no_grad():
            Q_targets = self.Q_target(next_states)
            Q_targets, _ = torch.max(Q_targets, -1, keepdim=True)
            Q_targets = rewards + self.gamma * (1 - dones) * Q_targets

        loss = (Q_values - Q_targets).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def initialize(self):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target_param.data.copy_(local_param.data)