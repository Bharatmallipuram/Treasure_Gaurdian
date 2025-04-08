# train_maddpg.py

import numpy as np
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env import TreasureGuardianEnv

# Keys for flattening
FIXED_KEYS = ["guardian", "keys", "pits", "treasure", "villains", "walls"]

def flatten_obs(obs):
    flat = []
    for k in FIXED_KEYS:
        flat.append(obs[k].flatten())
    return np.concatenate(flat)

Transition = namedtuple("Transition", 
                        ["state", "actions", "rewards", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    def add(self, s,a,r,ns,d):
        self.buffer.append(Transition(s,a,r,ns,d))
    def sample(self, bs):
        batch = random.sample(self.buffer, bs)
        batch = Transition(*zip(*batch))
        S = np.stack(batch.state)
        A = np.stack(batch.actions)
        R = np.stack(batch.rewards)
        NS= np.stack(batch.next_state)
        D = np.stack(batch.done)
        return S,A,R,NS,D
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, h=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.out = nn.Linear(h, act_dim)
    def forward(self, x, tau=1.0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return F.gumbel_softmax(logits, tau=tau, hard=True), logits

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, h=128):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.out = nn.Linear(h, 1)
    def forward(self, s, a):
        x = torch.cat([s,a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class MADDPGAgent:
    def __init__(self, idx, obs_dim, act_dim, state_dim, total_act_dim,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.95, tau=0.01, device='cpu'):
        self.idx = idx
        self.gamma = gamma
        self.tau   = tau
        self.device = device
        
        self.actor  = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(state_dim, total_act_dim).to(device)
        self.target_actor  = Actor(obs_dim, act_dim).to(device)
        self.target_critic = Critic(state_dim, total_act_dim).to(device)
        
        # copy weights
        self.target_actor .load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.a_opt = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, obs, tau=1.0):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if tau<=0:
                # greedy
                x = F.relu(self.actor.fc1(obs_t))
                x = F.relu(self.actor.fc2(x))
                logits = self.actor.out(x)
                idx = logits.argmax(dim=1).item()
                onehot = torch.zeros_like(logits)
                onehot[0,idx] = 1.0
                return idx, onehot.squeeze(0)
            else:
                onehot, _ = self.actor(obs_t, tau=tau)
                idx = onehot.argmax(dim=1).item()
                return idx, onehot.squeeze(0)
    
    def update(self, samples, all_agents, agent_i):
        S,A,R,NS,D = samples
        bs = S.shape[0]
        S  = torch.FloatTensor(S).to(self.device)
        NS = torch.FloatTensor(NS).to(self.device)
        D  = torch.FloatTensor(D).unsqueeze(1).to(self.device)
        R_i= torch.FloatTensor(R[:,agent_i]).unsqueeze(1).to(self.device)
        A_cat = torch.FloatTensor(A.reshape(bs,-1)).to(self.device)
        
        # critic target
        with torch.no_grad():
            next_acts = []
            for ag in all_agents:
                oh, _ = ag.target_actor(NS, tau=1.0)
                next_acts.append(oh)
            next_a_cat = torch.cat(next_acts, dim=1)
            Q_targ = self.target_critic(NS, next_a_cat)
            y = R_i + self.gamma * Q_targ * (1 - D)
        
        # critic loss
        Q_cur = self.critic(S, A_cat)
        c_loss = F.mse_loss(Q_cur, y)
        self.c_opt.zero_grad()
        c_loss.backward()
        self.c_opt.step()
        
        # actor loss
        split = list(torch.split(A_cat, self.actor.out.out_features, dim=1))
        # recompute local action
        oh, _ = self.actor(S, tau=1.0)
        split[agent_i] = oh
        a_mod = torch.cat(split, dim=1)
        a_loss = -self.critic(S, a_mod).mean()
        self.a_opt.zero_grad()
        a_loss.backward()
        self.a_opt.step()
        
        # soft updates
        for p,t in zip(self.actor.parameters(),  self.target_actor.parameters()):
            t.data.copy_(self.tau*p.data + (1-self.tau)*t.data)
        for p,t in zip(self.critic.parameters(), self.target_critic.parameters()):
            t.data.copy_(self.tau*p.data + (1-self.tau)*t.data)
        
        return c_loss.item(), a_loss.item()

def train_maddpg(env, n_episodes=500, max_steps=200, batch_size=128,
                 buffer_capacity=100000, update_every=100, device='cpu'):
    buf = ReplayBuffer(buffer_capacity)
    obs0 = env.reset()
    s0 = flatten_obs(obs0)
    print("State dim:", s0.shape)
    
    s_dim = s0.shape[0]
    a_dim = 4
    N = 1 + env.num_villains
    tot_a = N * a_dim
    
    agents = [MADDPGAgent(i, s_dim, a_dim, s_dim, tot_a, device=device)
              for i in range(N)]
    
    total_steps = 0
    for ep in range(n_episodes):
        obs = env.reset()
        state = flatten_obs(obs)
        rewards_ep = np.zeros(N)
        done=False; step=0
        
        while not done and step<max_steps:
            acts_int, acts_oh = [], []
            for ag in agents:
                ai, oh = ag.select_action(state, tau=1.0)
                acts_int.append(ai)
                acts_oh.append(oh.cpu().numpy())
            
            env_acts = {"guardian": acts_int[0],
                        "villains": np.array(acts_int[1:])}
            nxt, (g_r,v_r), done, info = env.step(env_acts)
            # split villain reward equally
            rs = [g_r] + [v_r/env.num_villains]*env.num_villains
            nxt_s = flatten_obs(nxt)
            buf.add(state, np.array(acts_oh), np.array(rs), nxt_s, done)
            
            state = nxt_s
            rewards_ep += np.array(rs)
            step +=1
            total_steps +=1
            
            if len(buf)>batch_size and total_steps%update_every==0:
                batch = buf.sample(batch_size)
                for idx, ag in enumerate(agents):
                    ag.update(batch, agents, idx)
        
        print(f"Ep {ep+1} | Reward: {rewards_ep} | Steps: {step} | Result: {info.get('result','n/a')}")
    return agents

if __name__=='__main__':
    env = TreasureGuardianEnv(render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained = train_maddpg(env, device=device)
    
    # --- testing ---
    obs = env.reset()
    st = flatten_obs(obs)
    done = False
    while not done:
        acts = []
        for ag in trained:
            ai, _ = ag.select_action(st, tau=0.0)   # greedy
            acts.append(ai)
        out, _, done, _ = env.step({"guardian":acts[0], "villains":np.array(acts[1:])})
        st = flatten_obs(out)
        env.render()
    env.close()
