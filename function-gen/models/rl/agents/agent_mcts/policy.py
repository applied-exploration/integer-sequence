import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils import dec2bin, BINARY_NUM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegerPolicy(nn.Module):
    """
    Simple neural network policy for solving the hill climbing task.
    Consists of one common dense layer for both policy and value estimate and
    another dense layer for each.
    """

    def __init__(self, n_obs, n_hidden, n_actions):
        super(IntegerPolicy, self).__init__()

        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_actions = n_actions

        self.dense1 = nn.Linear(n_obs, n_hidden)
        self.dense_p = nn.Linear(n_hidden, n_actions)
        self.dense_v = nn.Linear(n_hidden, 1)

    def forward(self, obs):
        # print("policy forward")
        # print(obs.shape[0])
        # print(self.n_obs)
        # print("obs")
        # print(obs)

        # obs_one_hot = torch.zeros((obs.shape[0], self.n_obs))
        # print("obs_one_hot")
        # print(obs_one_hot)
        # obs_one_hot[np.arange(obs.shape[0]), obs.numpy()] = 1.0
        # print(obs_one_hot)
        # h_relu = F.relu(self.dense1(obs_one_hot))

        obs = obs.to(torch.float)
        print(obs)
        obs_encoded = dec2bin(obs.squeeze(1).to(dtype=torch.long), BINARY_NUM).squeeze(1)
        print(obs_encoded)

        # print(obs.shape)
        h_relu = F.relu(self.dense1(obs))
        # print(h_relu.shape)
        logits = self.dense_p(h_relu)
        # print(logits.shape)
        policy = F.softmax(logits, dim=1)
        # print(policy.shape)

        value = self.dense_v(h_relu).view(-1)

        return logits, policy, value

    def step(self, obs):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        
        obs = np.array(obs, dtype=np.float32)
        obs = torch.from_numpy(obs)
        _, pi, v = self.forward(obs)

        return pi.detach().numpy(), v.detach().numpy()