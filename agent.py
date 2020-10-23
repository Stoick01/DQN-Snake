"""
Model input:
 transposed coded array of play area

Model utput:
 - 4 (l, r, u, d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy

import numpy as np

import csv
import math
from random import sample, random, choice

from typing import NamedTuple

class StepData(NamedTuple):
    state: list
    action: int
    reward: float
    next_state: list
    done: bool


class DQNAgent:
    def __init__(self, model, tgt):
        self.model = deepcopy(model)
        self.tgt = deepcopy(model)
        self.rb = ReplayBuffer()

        self.eps_decay = 0.9999994
        self.step_num = -10000
        self.model_update_cnt = -10000
        self.tgt_update_cnt = 0

        self.tgt_updated = False

        self.last_observation = None


    def update_model(self, model):
        self.model = deepcopy(model)
        self.model_update_cnt = 0

    def update_tgt(self, tgt):
        self.tgt = deepcopy(tgt)
        self.tgt_update_cnt = 0

    def restart_buffer(self):
        self.rb = ReplayBuffer()

    def get_action(self, observation):
        eps = self.eps_decay**(self.step_num)
        eps = max(eps, 0.1)
        self.last_observation = observation

        if random() < eps:
            self.action = choice([0, 1, 2, 3])
        else:
            self.action = self.model(torch.Tensor(observation)).max(-1)[-1].item()

        self.step_num += 1

        self.model_update_cnt += 1

        return self.action

    def step(self, reward, observation, done):
        sd = StepData(self.last_observation, self.action, reward, observation, done)

        self.rb.insert(sd)

        if self.rb.training_ready() and self.model_update_cnt % 80 == 0:
            self.model_update_cnt = 0
            self.tgt_update_cnt += 1
            self.model.train_step(self.rb.sample(2048), self.tgt)
            if self.tgt_update_cnt == 125:
                self.model.print_loss()
                self.tgt_update_cnt = 0
                self.tgt_updated = True
                self.tgt.upgrade_model(self.model)



class ReplayBuffer():
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.index = 0

        self.full = False

    def get_as_list(self):
        for el in self.buffer:
            if el == None:
                break
            arr = [el.state, el.action, el.reward, el.next_state, el.done]
            yield arr

    def training_ready(self):
        if self.full:
            return True
        
        if self.index >= 10000:
            return True
        return False

    def insert(self, data):
        """
        each element is StepData
        """

        self.buffer[self.index] = data

        self.index = (self.index + 1) % self.buffer_size

        if self.index == 0:
            self.full = True

    def sample(self, num_samples):
        """
        Returns random assortement of elements that are in buffer
        """
        if num_samples > self.index and not self.full:
            num_samples = self.index

        if self.index < self.buffer_size and not self.full:
            d = sample(self.buffer[:self.index], num_samples)
            return d

        d = sample(self.buffer[:], num_samples)
        return d

class Model(nn.Module):
    def __init__(self, obs_shape, num_actions):
        self.losses = []
        super(Model, self).__init__()

        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions),
            torch.nn.Softmax(dim=-1)
        )

        self.opt = optim.Adam(self.net.parameters(), lr=0.0001)

    def print_loss(self):
        s = 0
        for l in self.losses:
            s += l

        s /= len(self.losses)

        print('loss:', s)
        self.losses = []

    def forward(self, x):
        return self.net(x)

    def upgrade_model(self, m):
        """
        Updates target model with current models weights
        """
        self.net.load_state_dict(m.net.state_dict())

    def train_step(self, state_transitions, tgt):
        """
        Goes throught train step, wtih specific loss function designed for Q learning
        """

        current_states = torch.stack([torch.Tensor(s.state) for s in state_transitions])

        rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions])

        mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])

        next_state = torch.stack([torch.Tensor(s.next_state) for s in state_transitions])

        actions = [s.action for s in state_transitions]

        with torch.no_grad():
            qvals_next = tgt(next_state).max(-1)[0]

        self.opt.zero_grad()
        qvals = self(current_states) # (N, num_actions)

        one_hot_actions = F.one_hot(torch.LongTensor(actions), self.num_actions)
        
        loss = ((rewards[:, 0] + 0.95*mask[:, 0] * qvals_next - torch.sum(qvals * one_hot_actions, -1))**2).mean()
        self.losses.append(loss.item())

        loss.backward()
        self.opt.step()

        return loss