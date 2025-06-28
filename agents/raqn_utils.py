import random

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, sample):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(sample)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

import numpy as np

def compute_reward(state, next_state, weights=None):
    """
    计算状态转移带来的 reward。支持为不同指标分配不同权重。
    参数:
        state, next_state: numpy array 或 list
        weights: 可选 list/array，为每个状态维度指定权重
    返回:
        reward: float
    """
    state = np.array(state)
    next_state = np.array(next_state)
    diff = np.abs(next_state - state)

    if weights is not None:
        weights = np.array(weights)
        assert weights.shape == diff.shape, \
            f"weights.shape {weights.shape} != state.shape {diff.shape}"
        diff *= weights

    reward = -np.sum(diff)
    return reward

