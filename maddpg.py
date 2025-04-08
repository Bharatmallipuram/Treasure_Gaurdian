import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor Network
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)  # Discrete: logits

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # Logits

# Critic Network
class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, obs_all, actions_all):
        x = torch.cat([obs_all, actions_all], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, rewards, next_obs, dones):
        self.buffer.append((obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = map(np.array, zip(*batch))
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)

# Agent
class Agent:
    def __init__(self, obs_dim, action_dim, total_obs_dim, total_action_dim, lr=1e-3, tau=0.01, gamma=0.95):
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.target_actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(total_obs_dim, total_action_dim).to(device)
        self.target_critic = Critic(total_obs_dim, total_action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.tau = tau
        self.gamma = gamma
        self.action_dim = action_dim

        self.update_targets(hard=True)

    def select_action(self, obs, explore=True):
        obs = torch.FloatTensor(obs).to(device)
        logits = self.actor(obs)
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        if explore:
            action = np.random.choice(self.action_dim, p=probs)
        else:
            action = np.argmax(probs)
        return action

    def update_targets(self, hard=False):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data if hard else self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data if hard else self.tau * param.data + (1 - self.tau) * target_param.data)

# MADDPG Trainer
class MADDPG:
    def __init__(self, env, num_agents, buffer_size=100000, batch_size=1024, update_every=100):
        self.env = env
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.update_every = update_every
        self.steps = 0

        # ✅ Use proper keys based on observation space structure
        guardian_obs_dim = env.observation_space["guardian"].shape[0]
        villain_obs_dim = env.observation_space["villains"].shape[1]  # shape = (num_villains, obs_dim)
        guardian_act_dim = env.action_space["guardian"].n
        villain_act_dim = env.action_space["villains"].nvec[0]

        obs_dims = [guardian_obs_dim] + [villain_obs_dim] * (num_agents - 1)
        action_dims = [guardian_act_dim] + [villain_act_dim] * (num_agents - 1)

        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)

        self.agents = [
            Agent(obs_dims[i], action_dims[i], total_obs_dim, total_action_dim)
            for i in range(num_agents)
        ]

        self.buffer = ReplayBuffer(buffer_size)

        def act(self, obs, explore=True):
            """
            obs: List or array of observations for each agent, length = num_agents
            Returns: List of actions for each agent
            """
            actions = []
            for i, agent in enumerate(self.agents):
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                logits = agent.actor(obs_tensor)
                probs = torch.softmax(logits, dim=-1)

                if explore:
                    action = torch.multinomial(probs, 1).item()
                else:
                    action = torch.argmax(probs, dim=-1).item()

                actions.append(action)
            return actions


    def step(self, obs, actions, rewards, next_obs, dones):
        self.buffer.push(obs, actions, rewards, next_obs, dones)
        self.steps += 1

        if len(self.buffer) > self.batch_size and self.steps % self.update_every == 0:
            for agent_idx in range(self.num_agents):
                self.update(agent_idx)

    def update(self, agent_idx):
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs = torch.FloatTensor(obs).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        dones = torch.FloatTensor(dones).to(device)

        agent = self.agents[agent_idx]
        all_obs = obs.reshape(self.batch_size, -1)
        all_next_obs = next_obs.reshape(self.batch_size, -1)

        # Current joint action one-hot
        all_actions = []
        for i, a in enumerate(actions.T):
            one_hot = torch.nn.functional.one_hot(a, num_classes=self.agents[i].action_dim).float()
            all_actions.append(one_hot)
        all_actions_tensor = torch.cat(all_actions, dim=-1)

        # Next joint action (from target actors)
        all_next_actions = []
        for i, ag in enumerate(self.agents):
            logits = ag.target_actor(next_obs[:, i, :])
            probs = torch.softmax(logits, dim=-1)
            a = torch.multinomial(probs, 1).squeeze(1)
            one_hot = torch.nn.functional.one_hot(a, num_classes=ag.action_dim).float()
            all_next_actions.append(one_hot)
        all_next_actions_tensor = torch.cat(all_next_actions, dim=-1)

        # Critic loss
        with torch.no_grad():
            target_q = agent.target_critic(all_next_obs, all_next_actions_tensor)
            y = rewards[:, agent_idx].unsqueeze(1) + agent.gamma * target_q * (1 - dones[:, agent_idx].unsqueeze(1))

        current_q = agent.critic(all_obs, all_actions_tensor)
        critic_loss = nn.MSELoss()(current_q, y)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # Actor loss
        logits = agent.actor(obs[:, agent_idx, :])
        probs = torch.softmax(logits, dim=-1)
        sampled_action = torch.multinomial(probs, 1).squeeze(1)
        one_hot_action = torch.nn.functional.one_hot(sampled_action, num_classes=agent.action_dim).float()

        all_actions[agent_idx] = one_hot_action
        new_all_actions_tensor = torch.cat(all_actions, dim=-1)

        actor_loss = -agent.critic(all_obs, new_all_actions_tensor).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # Soft update targets
        agent.update_targets()

