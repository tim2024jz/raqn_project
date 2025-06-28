import torch, torch.nn as nn, torch.optim as optim, random, numpy as np
import gym

class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.mu_w=nn.Parameter(torch.FloatTensor(out_f,in_f));self.sigma_w=nn.Parameter(torch.FloatTensor(out_f,in_f))
        self.mu_b=nn.Parameter(torch.FloatTensor(out_f));self.sigma_b=nn.Parameter(torch.FloatTensor(out_f))
        self.reset_params()
    def reset_params(self):
        self.mu_w.data.normal_(0,0.1);self.sigma_w.data.fill_(0.017)
        self.mu_b.data.normal_(0,0.1);self.sigma_b.data.fill_(0.017)
    def forward(self,x):
        eps_w=torch.randn_like(self.sigma_w);eps_b=torch.randn_like(self.sigma_b)
        return nn.functional.linear(x,self.mu_w+self.sigma_w*eps_w,self.mu_b+self.sigma_b*eps_b)

class NoisyDQN(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.fc1=nn.Linear(state_dim,128);self.noisy=NoisyLinear(128,action_dim)
    def forward(self,x): return self.noisy(torch.relu(self.fc1(x)))

env=gym.make('CartPole-v1')
model,target=NoisyDQN(4,10),NoisyDQN(4,10);target.load_state_dict(model.state_dict())
opt=optim.Adam(model.parameters(),1e-2);memory=[];gamma=0.99

for ep in range(500):
    state,done=env.reset(),False
    while not done:
        with torch.no_grad():q=model(torch.FloatTensor(state));action=torch.argmax(q).item()
        next_state,reward,done,_=env.step(action);memory.append((state,action,reward,next_state))
        if len(memory)>6000:memory.pop(0);state=next_state
        if len(memory)<32:continue
        s,a,r,s1=zip(*random.sample(memory,32))
        s,a,r,s1=torch.FloatTensor(s),torch.LongTensor(a).unsqueeze(1),torch.FloatTensor(r).unsqueeze(1),torch.FloatTensor(s1)
        q=model(s).gather(1,a);q1=target(s1).max(1)[0].detach().unsqueeze(1)
        loss=nn.MSELoss()(q,r+gamma*q1)
        opt.zero_grad();loss.backward();opt.step()
    if ep%10==0:target.load_state_dict(model.state_dict())
