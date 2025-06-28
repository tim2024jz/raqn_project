import torch
import torch.optim as optim
import random
import numpy as np
from agents.raqn_model import RAQNNet
from agents.aco_lb_guidance import generate_guided_sample
from agents.raqn_utils import ReplayMemory, compute_reward
import gym

# === 配置超参数 ===
state_dim, action_dim = 4, 10
gamma = 0.9
epsilon, min_epsilon, decay = 1.0, 0.05, 0.995
eta, min_eta, eta_decay = 0.95, 0.05, 0.995
batch_size, sync_interval = 32, 200
num_episodes, max_steps, memory_capacity = 500, 200, 6000
lr = 1e-2

# === 环境 & 设备 ===
env = gym.make('CartPole-v1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 初始化网络与优化器 ===
policy_net = RAQNNet(state_dim, action_dim).to(device)
target_net = RAQNNet(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # 目标网络设置为 eval 模式

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = ReplayMemory(memory_capacity)

global_step = 0

for episode in range(num_episodes):
    raw_state = env.reset()
    state = np.zeros(state_dim)  # 这里应替换为 SD-IoT 的状态编码逻辑
    done = False

    for step in range(max_steps):
        global_step += 1

        # === 动作选择 ===
        if random.random() < eta:
            guided_sample = generate_guided_sample(state)
            memory.push(guided_sample)
            state = guided_sample[3]  # guided_sample = (state, action, reward, next_state)
        else:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()

            next_raw_state, reward, done, _ = env.step(action)
            next_state = np.zeros(state_dim)  #  
            reward = compute_reward(state, next_state)   
            memory.push((state, action, reward, next_state))
            state = next_state

        # === 参数衰减 ===
        epsilon = max(min_epsilon, epsilon * decay)
        eta = max(min_eta, eta * eta_decay)

        # === 更新策略网络 ===
        if len(memory) >= batch_size:
            transitions = memory.sample(batch_size)
            states, actions, rewards, next_states = zip(*transitions)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)

            current_q = policy_net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + gamma * max_next_q

            loss = torch.nn.functional.mse_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # === 同步目标网络 ===
        if global_step % sync_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    print(f"Episode {episode + 1}/{num_episodes} finished after {step + 1} steps.")

env.close()
