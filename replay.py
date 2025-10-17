import numpy as np
import torch
from collections import deque

class RingReplay:
    def __init__(self, state_dim, capacity=200_000, device="cpu"):
        self.s  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a  = np.zeros((capacity, 1),         dtype=np.int64)   # {0,1,2}
        self.r  = np.zeros((capacity,),           dtype=np.float32)
        self.sn = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d  = np.zeros((capacity,),           dtype=np.float32)
        self.idx = 0
        self.full = False
        self.cap = capacity
        self.device = device

    def append(self, s, a_idx, r, sn, d):
        i = self.idx
        self.s[i]  = s
        self.a[i,0]= a_idx
        self.r[i]  = r
        self.sn[i] = sn
        self.d[i]  = d
        self.idx = (i + 1) % self.cap
        self.full = self.full or (self.idx == 0)

    def __len__(self):
        return self.cap if self.full else self.idx

    def sample(self, batch):
        m = len(self)
        idx = np.random.randint(0, m, size=(batch,))
        s  = torch.as_tensor(self.s[idx],  device=self.device)
        a  = torch.as_tensor(self.a[idx],  device=self.device)
        r  = torch.as_tensor(self.r[idx],  device=self.device)
        sn = torch.as_tensor(self.sn[idx], device=self.device)
        d  = torch.as_tensor(self.d[idx],  device=self.device)
        return s, a, r, sn, d

class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.traj = deque(maxlen=n)

    def push(self, s, a, r):
        self.traj.append((s, a, r))

    def flush(self, sn, done):
        if len(self.traj) == 0:
            return None
        R = 0.0
        for i, (_, _, r_i) in enumerate(self.traj):
            R += (self.gamma ** i) * float(r_i)
        s0, a0, _ = self.traj[0]
        self.traj.clear()
        return (s0, a0, R, sn, done)
