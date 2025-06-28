import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_):
        self.buffer.append((s, a, r, s_))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_ = map(np.stack, zip(*batch))
        return s, a, r, s_

    def __len__(self):
        return len(self.buffer)
