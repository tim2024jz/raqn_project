import torch, torch.nn as nn, torch.optim as optim, gym, numpy as np

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(state_dim,128), nn.ReLU(), nn.Linear(128,action_dim))
    def forward(self,x): return self.fc(x)

env=gym.make('CartPole-v1')
policy=PolicyNet(4,10)
opt=optim.Adam(policy.parameters(),lr=3e-4)
eps_clip=0.2;gamma=0.99

for ep in range(500):
    state=env.reset();done=False;log_probs=[];rewards=[];states=[];actions=[]
    while not done:
        s=torch.FloatTensor(state)
        logits=policy(s)
        prob=torch.softmax(logits,dim=-1)
        action=torch.multinomial(prob,1).item()
        log_prob=torch.log(prob[action])
        next_state, reward, done,_=env.step(action)
        log_probs.append(log_prob);rewards.append(reward);states.append(s);actions.append(action)
        state=next_state
    returns=[];G=0
    for r in reversed(rewards): G=r+gamma*G;returns.insert(0,G)
    returns=torch.FloatTensor(returns)
    log_probs=torch.stack(log_probs)
    ratio=torch.exp(log_probs-log_probs.detach())
    advantage=returns-returns.mean()
    surr1=ratio*advantage
    surr2=torch.clamp(ratio,1-eps_clip,1+eps_clip)*advantage
    loss=-torch.min(surr1,surr2).mean()
    opt.zero_grad();loss.backward();opt.step()
