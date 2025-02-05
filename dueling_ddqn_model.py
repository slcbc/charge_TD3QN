"""
This file defines the NN used by D3QN agent.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        torch.nn.init.constant_(m.bias, 0)


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, n_actions, s_dim):
        super(DuelingDeepQNetwork, self).__init__()

        st_dim,  sn_dim = s_dim

        # Q1 architecture
        n1, n2, n3, n4 = 256,256,128,64
        self.linear1 = nn.Linear(st_dim + sn_dim, n1)

        # self.bn1 = nn.BatchNorm1d(n1)
        # self.linear1 = nn.Linear(st_dim + sn_dim, 256)
        # self.linear2 = nn.Linear(n1, n2)
        # self.linear3 = nn.Linear(n2, n3)
        # self.linear4 = nn.Linear(n3, n4)

        n_out = n2
        self.V = nn.Linear(n_out, 1)
        self.A = nn.Linear(n_out, n_actions)


        self.apply(weights_init_)

    def forward(self, state):
        st, sn = state
        state_new = torch.cat([st,sn],dim=1)
        # state_new = torch.cat([st, sn], dim=1)
        # print(state_new.shape)
        x1 = self.linear1(state_new)
        # print('b',x1.shape)
        # x1 = self.bn1(x1)
        x1 = F.relu(x1)
        # x1 = self.linear2(x1)
        # x1 = F.relu(x1)
        # x1 = self.linear3(x1)
        # x1 = F.relu(x1)

        V = self.V(x1)
        A = self.A(x1)

        return V, A

    # def save_checkpoint(self):
    #     print('... saving checkpoint ...')
    #     T.save(self.state_dict(), self.checkpoint_file)
    #
    # def load_checkpoint(self):
    #     print('... loading checkpoint ...')
    #     self.load_state_dict(T.load(self.checkpoint_file))