"""
This is the file used to save the transitions (s,a,r,s_,done).
"""
import numpy as np
import random


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        l = len(self.buffer)
        min_batch_size = np.minimum(batch_size, l)
        batch = random.sample(self.buffer, min_batch_size)
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))
        st_batch, sn_batch = [], []
        st_batch_, sn_batch_, = [], []
        action, reward, done = [], [], []
        for i in batch:
            st, sn = i[0]
            action.append(i[1])
            reward.append(i[2])
            st_, sn_ = i[3]
            done.append(i[4])
            st_batch.append(st)
            sn_batch.append(sn)
            st_batch_.append(st_)
            sn_batch_.append(sn_)
        return [st_batch, sn_batch], action, reward, [st_batch_, sn_batch_], done

    def __len__(self):
        return len(self.buffer)