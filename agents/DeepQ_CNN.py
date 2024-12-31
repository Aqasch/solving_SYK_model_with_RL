import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import namedtuple, deque
import numpy as np
from itertools import product

from utils import dictionary_of_actions, dict_of_actions_revert_q


class DQN_CNN(object):

    def __init__(self, conf, action_size, state_size, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        # network_dim = conf['agent']['dimension']
        memory_size = conf['agent']['memory_size']
        
        self.final_gamma = conf['agent']['final_gamma']
        self.epsilon_min = conf['agent']['epsilon_min']
        self.epsilon_decay = conf['agent']['epsilon_decay']
        learning_rate = conf['agent']['learning_rate']
        self.update_target_net = conf['agent']['update_target_net']
        self.layers = int(conf['agent']['cnn_layer'])
        channels = []
        for l in range(self.layers):
            channels.append(int(conf['agent'][f'channel{l+1}']))
        # print(channels)
        # exit()

        drop_prob = conf['agent']['dropout']
        self.with_angles = conf['agent']['angles']
        
        if "memory_reset_switch" in conf['agent'].keys():
            self.memory_reset_switch =  conf['agent']["memory_reset_switch"]
            self.memory_reset_threshold = conf['agent']["memory_reset_threshold"]
            self.memory_reset_counter = 0
        else:
            self.memory_reset_switch =  False
            self.memory_reset_threshold = False
            self.memory_reset_counter = False

        self.action_size = action_size

        self.state_size = state_size if self.with_angles else state_size - self.num_layers*self.num_qubits*3

        self.state_size = self.state_size + 1 if conf['agent']['en_state'] else self.state_size
        self.state_size = self.state_size + 1 if ("threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]) else self.state_size

        self.translate = dictionary_of_actions(self.num_qubits)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)
        self.policy_net = self.unpack_network(channels, drop_prob).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        print(self.policy_net)
        # print(self.policy_net(torch.zeros(1, self.state_size)).view(1, -1).size(1))
        # print(self.target_net)
        

        self.gamma = torch.Tensor([np.round(np.power(self.final_gamma,1/self.num_layers),2)]).to(device)   
        self.memory = ReplayMemory(memory_size)
        self.epsilon = 1.0  

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()
        self.device = device
        self.step_counter = 0

   
        self.Transition = namedtuple('Transition',
                            ('state', 'action', 'reward',
                            'next_state','done'))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state, ill_action):
        state = state.unsqueeze(0)
        epsilon = False
        
        if torch.rand(1).item() <= self.epsilon:
            rand_ac = torch.randint(self.action_size, (1,)).item()
            while rand_ac in ill_action:
                rand_ac = torch.randint(self.action_size, (1,)).item()
            epsilon = True
            return (rand_ac, epsilon)
        act_values = self.policy_net.forward(state)
        act_values[0][ill_action] = float('-inf') 

        return torch.argmax(act_values[0]).item(), epsilon

    def replay(self, batch_size):
        if self.step_counter %self.update_target_net ==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1
        
        transitions = self.memory.sample(batch_size)
        batch = self.Transition(*zip(*transitions))
        # print(self.state_size, self.action_size)
        # exit()
        next_state_batch = torch.stack(batch.next_state)
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        done_batch = torch.stack(batch.done)

        # print(state_batch.shape)
        
        # print(state_batch.shape[0], state_batch.shape[1], 2,2,2)
        # state_batch = [state_batch.shape[0], state_batch.shape[1], 2,2,2]
        state_action_values = self.policy_net.forward(state_batch).gather(1, action_batch.unsqueeze(1))
        """ Double DQN """        
        next_state_values = self.target_net.forward(next_state_batch)
        next_state_actions = self.policy_net.forward(next_state_batch).max(1)[1].detach()
        next_state_values = next_state_values.gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
        
       
    
        """ Compute the expected Q values """
        expected_state_action_values = (next_state_values * self.gamma) * (1-done_batch) + reward_batch
        expected_state_action_values = expected_state_action_values.view(-1, 1)

        assert state_action_values.shape == expected_state_action_values.shape, "Wrong shapes in loss"
        cost = self.fit(state_action_values, expected_state_action_values)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon,self.epsilon_min)
        assert self.epsilon >= self.epsilon_min, "Problem with epsilons"
        return cost

    def fit(self, output, target_f):
        self.optim.zero_grad()
        loss = self.loss(output, target_f)
        loss.backward()
        self.optim.step()
        return loss.item()

    def unpack_network(self, channels, p):
        kernel_size, padding, stride = (3, 3, 3), (1, 1, 1), (2, 2, 2)
        kernel_size_pool = (1, 1, 1)
        in_channel = 1
        layer = []
        for channel in channels:
            layer.append(nn.Conv3d(in_channel, channel, kernel_size= kernel_size, padding=padding))
            layer.append(nn.LeakyReLU())
            layer.append(nn.MaxPool3d(kernel_size=kernel_size_pool, stride=stride))
            in_channel = channel
        
        # Flatten output tensor
        layer.append(nn.Flatten())
        # Dense layers
        if self.num_qubits ==4:
            if self.layers == 3:
                in_feat = 640
            elif self.layers == 4:
                in_feat = 640+128
            elif self.layers == 5:
                in_feat  = 640+3*128
            elif self.layers == 6:
                in_feat  = 640+3*128
            elif self.layers == 7:
                in_feat  = 640+11*128
        if self.num_qubits ==6:
            if self.layers == 3:
                in_feat = 2304
            elif self.layers == 4:
                # in_feat = 1280 # WITHOUT MODD!!!
                # in_feat = 2560 # WITH MOD 1
                in_feat = 5120 # WITH MOD 2
            elif self.layers == 5:
                in_feat  = 1536
            elif self.layers == 6:
                in_feat  = 2048
            elif self.layers == 7:
                in_feat  = 2048
        layer.append(nn.Linear(in_feat, in_feat//2))
        layer.append(nn.LeakyReLU())
        layer.append(nn.Linear(in_feat//2, self.action_size))
    
        return nn.Sequential(*layer)
    
    # def unpack_network(self, channels, kernel_sizes, strides, p):
    #     layers = []
    #     input_channels = 1  # Number of input channels
    #     channels = [32,64,128]
    #     kernel_size = (3,3,3)
    #     for output_channels in channels:
    #         layers.append(nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size, padding=(1, 1, 1)))
    #         layers.append(nn.LeakyReLU())  # Using Leaky ReLU
    #         layers.append(nn.MaxPool3d(kernel_size=(2,2,2), stride = (2,2,2)))  # Optional: Add max pooling layer
    #         input_channels = output_channels
    #     layers.append(nn.Flatten())
    #     layers.append(nn.Linear(1280, 512))
    #     layers.append(nn.LeakyReLU())
    #     layers.append(nn.Linear(512, self.action_size))
    #     return nn.Sequential(*layers)
    


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                                    ('state', 'action', 'reward',
                                    'next_state','done'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def clean_memory(self):
        self.memory = []
        self.position = 0

if __name__ == '__main__':
    pass