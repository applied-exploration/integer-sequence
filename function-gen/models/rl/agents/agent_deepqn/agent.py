import numpy as np
import random
from collections import namedtuple, deque

import sys, os
sys.path.append(os.path.abspath('..'))

from .model import QNetwork
from models.rl.agents.abstract_agent import Agent
from utils import flatten

import torch
import torch.nn.functional as F
import torch.optim as optim

import uuid
import time

class DeepQAgentConfig:

    def __init__(self,   
                BUFFER_SIZE =  int(1e5),  # replay buffer size
                BATCH_SIZE = 64,         # minibatch size
                GAMMA = 0.99,            # discount factor
                TAU = 1e-3,              # for soft update of target parameters
                LR = 5e-4,               # learning rate 
                UPDATE_EVERY = 4,       # how often to update the network)
                HIDDEN_LAYER_SIZE = [32, 32]):
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR = LR
        self.UPDATE_EVERY = UPDATE_EVERY
        self.HIDDEN_LAYER_SIZE = HIDDEN_LAYER_SIZE


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeepQAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config = DeepQAgentConfig(), seed = 1, samp_frames=1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.config = config

        # Q-Network
        self.qnetwork_local = QNetwork(state_size *samp_frames, action_size, seed, hidden_layer_param = self.config.HIDDEN_LAYER_SIZE).to(device)
        self.qnetwork_target = QNetwork(state_size *samp_frames, action_size, seed, hidden_layer_param = self.config.HIDDEN_LAYER_SIZE).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config.LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.config.BUFFER_SIZE, self.config.BATCH_SIZE, seed, samp_frames)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.id = uuid.uuid4()

    def get_title(self):
        for_id = "{} \n {}".format(time.strftime("%Y-%m-%d_%H%M%S"), self.id)
        for_title = "DQN with Hidden layers: {}".format(' '.join([str(elem) for elem in self.config.HIDDEN_LAYER_SIZE]))
        for_filename = "DQN_Network size_{}".format(' '.join([str(elem) for elem in self.config.HIDDEN_LAYER_SIZE]))
        for_table = [['hidden layers', 'learning rate', 'buffer size', 'batch size'],[[' '.join([str(elem) for elem in self.config.HIDDEN_LAYER_SIZE])], [self.config.LR], [self.config.BUFFER_SIZE], [self.config.BATCH_SIZE] ]]
        return for_title, for_filename, for_table, for_id

    def save(self, experiment_num, num_agent):
        torch.save(self.qnetwork_local.state_dict(), 'experiments/trained_agents/{}_dqn_exp{}__agent{}_{}.pth'.format(time.strftime(
            "%Y-%m-%d"),experiment_num, num_agent, self.id))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.BATCH_SIZE:
                experiences = self.memory.sample()
                self.__learn(experiences, self.config.GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def reset(self):
        pass

    def __learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.__soft_update(self.qnetwork_local, self.qnetwork_target, self.config.TAU)                     

    def __soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)





class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, samp_frames):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.samp_frames = samp_frames
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        if self.samp_frames > 1:
            experience_ids = random.sample(np.arange(0,len(self.memory)-1,1).tolist(), k=int(self.batch_size/self.samp_frames))

            states = torch.from_numpy(np.vstack([flatten([self.memory[i].state, self.memory[i+1].state]) for i in experience_ids if self.memory[i] is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([self.memory[i].action for i in experience_ids if self.memory[i] is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([np.mean([self.memory[i].reward, self.memory[i+1].reward]) for i in experience_ids if self.memory[i] is not None])).float().to(device)
            # rewards = torch.from_numpy(np.vstack([np.sum([self.memory[i].reward, self.memory[i+1].reward]) for i in experience_ids if self.memory[i] is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([flatten([self.memory[i].next_state, self.memory[i+1].next_state]) for i in experience_ids if self.memory[i] is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([self.memory[i].done for i in experience_ids if self.memory[i] is not None]).astype(np.uint8)).float().to(device)

        else:
            experiences = random.sample(self.memory, k=self.batch_size)

            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        e = (states, actions, rewards, next_states, dones)

        # print(states[0][:4]*8)

        # print(states[0][4:]*8)
        # print("=====")
        return e
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)