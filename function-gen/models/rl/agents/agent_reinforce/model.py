
import gym
import numpy as np
from collections import deque

import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, state_size, action_size, seed=1, hidden_layer_param=[]):
        super(Model, self).__init__()

        self.seed = torch.manual_seed(seed)

        new_hidden_layer_param = hidden_layer_param.copy()

         # --- Input layer --- #
        self.fc_in = nn.Linear(state_size, new_hidden_layer_param[0])

        # --- Hidden layers --- #
        if len(new_hidden_layer_param) < 2: self.hidden_layers = []
        else: self.hidden_layers = nn.ModuleList([nn.Linear(new_hidden_layer_param[i], new_hidden_layer_param[i+1]) for i in range(len(new_hidden_layer_param)-1)])

        # --- Output layer --- #
        self.fc_out = nn.Linear(new_hidden_layer_param[-1], action_size)


    def forward(self, state):

        x = F.relu(self.fc_in(state))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        return F.softmax(self.fc_out(x), dim=1)

    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)