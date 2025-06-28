import torch
import torch.optim as optim
import random
import numpy as np
from agents.raqn_model import RAQNNet  # 复用RAQN网络结构
import gym

state_dim, action_dim = 4, 10
env = gym.make('CartPole-v1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RAQNNet(state_dim, action_dim).to(device)
target_model = RAQNNet(state_dim, action_dim).to(device)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=1e-2)
memory = []

gamma, epsilon, min_epsilon, decay = 0.9, 1.0, 0.05, 0.995
batch_size, sync_interval = 32, 200
num_episodes, max_steps = 500, 200

for episode in range(num_episodes):
    state, done = env.reset(), False
    state = np.zeros(state_dim)

    for t in range(max_steps):
        if random.random() < epsilon:
            action = random.randint(0, action_dim-1)
        else:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).to(device))
                action = torch.argmax(q_values).item()

        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state))
        if len(memory) > 6000:
            memory.pop(0)
        state = next_state

        if len(memory) >= batch_size:
            samples = random.sample(memory, batch_size)
            states, actions, rewards, next_states = zip(*samples)
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)

            q_values = model(states).gather(1, actions)
            next_q_values = target_model(next_states).max(1)[0].detach().unsqueeze(1)
            target_q = rewards + gamma * next_q_values

            loss = torch.nn.functional.mse_loss(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t % sync_interval == 0:
            target_model.load_state_dict(model.state_dict())

        if done:
            break

    if epsilon > min_epsilon:
        epsilon *= decay
