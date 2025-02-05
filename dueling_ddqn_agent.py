"""
This file defines the  D3QN agent.
"""

import numpy as np
import torch
import torch.nn.functional as F
from charge.D3QN_charge.dueling_ddqn_model import DuelingDeepQNetwork
from torch.optim import Adam
import os
import math

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DuelingDDQNAgent(object):
    def __init__(self, s_dim, num_actions,  args):
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.n_actions = num_actions
        self.input_dims = s_dim
        self.batch_size = args.batch_size
        self.eps_min = args.eps_min
        self.eps_dec = args.eps_dec
        self.target_update_interval = args.target_update_interval
        # self.algo = algo

        self.action_space = [i for i in range(num_actions)]
        self.learn_step_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.q1_eval = DuelingDeepQNetwork(self.n_actions, s_dim=self.input_dims).to(self.device)
        self.q2_eval = DuelingDeepQNetwork(self.n_actions, s_dim=self.input_dims).to(self.device)
        self.q1_next = DuelingDeepQNetwork(self.n_actions, s_dim=self.input_dims).to(self.device)
        self.q2_next = DuelingDeepQNetwork(self.n_actions, s_dim=self.input_dims).to(self.device)
        self.q1_eval_optim = Adam(self.q1_eval.parameters(), lr=args.lr, eps=1e-4,weight_decay=1e-4)
        self.q2_eval_optim = Adam(self.q2_eval.parameters(), lr=args.lr, eps=1e-4, weight_decay=1e-4)

        hard_update(self.q1_next, self.q1_eval)
        hard_update(self.q2_next, self.q2_eval)


    def choose_action(self, state,train):
        if np.random.random() < self.epsilon and train:
            act_index = np.random.randint(low=0, high=self.n_actions)
            return act_index

        else:
            st,  sn = state
            st = torch.FloatTensor(st).to(self.device).unsqueeze(0)
            sn = torch.FloatTensor(sn).to(self.device).unsqueeze(0)
            _, advantages1 = self.q1_eval.forward([st, sn])
            # _, advantages2 = self.q2_eval.forward([st, sn])
            # print('adv',advantages)
            act_index = torch.argmax(advantages1).item()
            # act_index2 = torch.argmax(advantages2).item()
            # print(act_index1)

            return act_index


    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def learn(self,memory, batch_size, updates):   # replay_buffer,self.batch_size,total_steps
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        st_batch, sn_batch = state_batch
        st_batch = torch.FloatTensor(st_batch).to(self.device)
        sn_batch = torch.FloatTensor(sn_batch).to(self.device)
        state_batch = [st_batch,sn_batch]

        next_st_batch, next_sn_batch = next_state_batch
        next_st_batch = torch.FloatTensor(next_st_batch).to(self.device)
        next_sn_batch = torch.FloatTensor(next_sn_batch).to(self.device)
        next_state_batch = [next_st_batch, next_sn_batch]

        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.LongTensor(mask_batch).to(self.device)

        indices = np.arange(batch_size)

        with torch.no_grad():
            V1_s_, A1_s_ = self.q1_next.forward(next_state_batch)
            V2_s_, A2_s_ = self.q2_next.forward(next_state_batch)
            q1_next = torch.add(V1_s_, (A1_s_ - A1_s_.mean(dim=1, keepdim=True)))
            q2_next = torch.add(V2_s_, (A2_s_ - A2_s_.mean(dim=1, keepdim=True)))
            # q_eval是current,q_next是target
            V1_s_eval, A1_s_eval = self.q1_eval.forward(next_state_batch)
            V2_s_eval, A2_s_eval = self.q2_eval.forward(next_state_batch)
            q1_next_eval = torch.add(V1_s_eval, (A1_s_eval - A1_s_eval.mean(dim=1,keepdim=True)))
            q2_next_eval = torch.add(V2_s_eval, (A2_s_eval - A2_s_eval.mean(dim=1, keepdim=True)))
            max1_actions_next = torch.argmax(q1_next_eval, dim=1)
            max2_actions_next = torch.argmax(q2_next_eval, dim=1)

        V1_s, A1_s = self.q1_eval.forward(state_batch)
        V2_s, A2_s = self.q2_eval.forward(state_batch)
        q1_pred = torch.add(V1_s, (A1_s - A1_s.mean(dim=1, keepdim=True)))[indices, action_batch]
        q2_pred = torch.add(V2_s, (A2_s - A2_s.mean(dim=1, keepdim=True)))[indices, action_batch]
        q1_next[mask_batch] = 0.0
        q2_next[mask_batch] = 0.0
        q1_target_values = q1_next[indices, max1_actions_next]
        q2_target_values = q2_next[indices, max2_actions_next]
        q_target_values = torch.min(q1_target_values,q2_target_values)
        q_target = reward_batch + self.gamma*q_target_values

        # loss = torch.nn.MSELoss(q_target, q_pred).to(self.device)
        loss1 = F.mse_loss(q1_pred, q_target).to(self.device)
        loss2 = F.mse_loss(q2_pred, q_target).to(self.device)
        self.q1_eval_optim.zero_grad()
        self.q2_eval_optim.zero_grad()
        loss1.backward()
        loss2.backward()
        self.q1_eval_optim.step()
        self.q2_eval_optim.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        if updates % self.target_update_interval == 0:
            hard_update(self.q1_next, self.q1_eval)
            hard_update(self.q2_next, self.q2_eval)

    def save_model(self, name="", sg=None, set=None, suffix="", eval_path = None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if eval_path is None:
            eval_path = "models/sg{}/Qeval_{}{}{}".format(sg, name, set, suffix)
            # eval_path2 = "models/st{}/Qeval2_{}{}{}".format(sg, name, set, suffix)
        torch.save(self.q1_eval.state_dict(), eval_path)
        # torch.save(self.q2_eval.state_dict(), eval_path2)

        print('Q_eval model parameters are saved to {}'.format(eval_path))

    # Load model parameters
    def load_model(self, name="", sg=None, set=None, suffix="", eval_path = None):
        if eval_path is None:
            eval_path = "models/sg{}/Qeval_{}{}{}".format(sg, name, set, suffix)
            # eval_path2 = "models/st{}/Qeval2_{}{}{}".format(sg, name, set, suffix)

        print('Loading models from {}'.format(eval_path))
        if eval_path is not None:
            self.q1_eval.load_state_dict(torch.load(eval_path))
            # self.q2_eval.load_state_dict(torch.load(eval_path2))
