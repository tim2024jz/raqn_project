import torch, torch.nn as nn, torch.optim as optim, random, numpy as np
import gym

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(state_dim,128),nn.ReLU(),nn.Linear(128,action_dim),nn.Tanh())
    def forward(self,x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(state_dim+action_dim,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,s,a): return self.net(torch.cat([s,a],dim=-1))

env=gym.make('Pendulum-v1')
state_dim,action_dim=env.observation_space.shape[0],env.action_space.shape[0]
actor,critic=Actor(state_dim,action_dim),Critic(state_dim,action_dim)
optA,optC=optim.Adam(actor.parameters(),1e-3),optim.Adam(critic.parameters(),1e-3)
targetA,targetC=Actor(state_dim,action_dim),Critic(state_dim,action_dim)
targetA.load_state_dict(actor.state_dict());targetC.load_state_dict(critic.state_dict())
gamma, tau, memory=[],0.005,[]

for ep in range(300):
    state,_=env.reset();done=False
    while not done:
        action=actor(torch.FloatTensor(state)).detach().numpy()
        next_state, reward, done,_=env.step(action+np.random.normal(0,0.1,action_dim))
        memory.append((state,action,reward,next_state,done))
        if len(memory)>10000:memory.pop(0)
        state=next_state
        if len(memory)<64:continue
        s,a,r,s1,d=zip(*random.sample(memory,64))
        s,a,r,s1=torch.FloatTensor(s),torch.FloatTensor(a),torch.FloatTensor(r).unsqueeze(1),torch.FloatTensor(s1)
        y=r+gamma*targetC(s1,targetA(s1)).detach()
        critic_loss=nn.MSELoss()(critic(s,a),y)
        optC.zero_grad();critic_loss.backward();optC.step()
        actor_loss=-critic(s,actor(s)).mean()
        optA.zero_grad();actor_loss.backward();optA.step()
        for param, target_param in zip(actor.parameters(),targetA.parameters()): target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
        for param, target_param in zip(critic.parameters(),targetC.parameters()): target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
