import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import RAQNNet
from config import CONFIG

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])

        return states, actions, rewards, next_states, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


def train_per_dqn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAQNNet(CONFIG['input_dim'], CONFIG['output_dim']).to(device)
    target = RAQNNet(CONFIG['input_dim'], CONFIG['output_dim']).to(device)
    target.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['alpha'])
    buffer = PrioritizedReplayBuffer(CONFIG['buffer_size'])
    epsilon = CONFIG['epsilon_start']
    beta = 0.4
    gamma = CONFIG['gamma']
    rewards = []

    for ep in range(CONFIG['num_episodes']):
        state = np.random.rand(CONFIG['input_dim'])
        done = False
        episode_reward = 0
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(CONFIG['output_dim'])
            else:
                with torch.no_grad():
                    q_vals = model(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = q_vals.argmax().item()
            next_state = state + np.random.normal(0, 0.05, CONFIG['input_dim'])
            reward = state[0] - 0.3*state[1] - 0.2*state[2] - 0.2*state[3]
            buffer.push(state, action, reward, next_state)
            state = next_state
            episode_reward += reward

            if len(buffer.buffer) >= CONFIG['batch_size']:
                s, a, r, s_, idxs, weights = buffer.sample(CONFIG['batch_size'], beta)
                s = torch.FloatTensor(s).to(device)
                a = torch.LongTensor(a).unsqueeze(1).to(device)
                r = torch.FloatTensor(r).unsqueeze(1).to(device)
                s_ = torch.FloatTensor(s_).to(device)
                weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

                q_vals = model(s).gather(1, a)
                next_q = target(s_).max(1)[0].unsqueeze(1)
                targets = r + gamma * next_q

                td_errors = (q_vals - targets).detach().cpu().numpy()
                loss = (weights * (q_vals - targets) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                buffer.update_priorities(idxs, np.abs(td_errors) + 1e-5)

        rewards.append(episode_reward)
        if ep % CONFIG['sync_interval'] == 0:
            target.load_state_dict(model.state_dict())
        epsilon = max(CONFIG['epsilon_end'], epsilon * 0.995)
        beta = min(1.0, beta + 0.001)
    np.save('logs/per_dqn_rewards.npy', rewards)

if __name__ == '__main__':
    train_per_dqn()
