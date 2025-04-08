import numpy as np
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import your environment from env.py
from old_env import TreasureGuardianEnv


# Fixed Keys and Helper: Flatten the Environment Observation

# Adjust FIXED_KEYS to match exactly what your env.py returns.
FIXED_KEYS = ["guardian", "keys", "pits", "treasure", "villains", "walls"]
# If your environment now uses "pits" instead of "snakes", change the above accordingly.

def flatten_obs(obs):
    """
    Flatten the observation dictionary into a 1D numpy array using a fixed key order.
    """
    flat_list = []
    for key in FIXED_KEYS:
        if key not in obs:
            raise ValueError(f"Expected key '{key}' in observation, but it was not found.")
        flat_list.append(obs[key].flatten())
    return np.concatenate(flat_list)


# Replay Buffer for Multi-Agent Experiences

Transition = namedtuple("Transition", 
                        ["state", "actions", "rewards", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, actions, rewards, next_state, done):
        self.buffer.append(Transition(state, actions, rewards, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        states = np.stack(batch.state)
        actions = np.stack(batch.actions)
        rewards = np.stack(batch.rewards)
        next_states = np.stack(batch.next_state)
        dones = np.stack(batch.done)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# Neural Network Definitions


# Actor Network: Maps local observation to a probability distribution over actions.
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, act_dim)
    
    def forward(self, obs, tau=1.0):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        # Use Gumbel-softmax to get a one-hot vector for discrete actions.
        action = F.gumbel_softmax(logits, tau=tau, hard=True)
        return action

# Critic Network: Centralized critic that takes the global state and all agents' actions.
class Critic(nn.Module):
    def __init__(self, state_dim, total_act_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + total_act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value


# MADDPG Agent Definition

class MADDPGAgent:
    def __init__(self, agent_id, obs_dim, act_dim, state_dim, total_act_dim,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.95, tau=0.01, device='cpu'):
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(state_dim, total_act_dim).to(device)
        self.target_actor = Actor(obs_dim, act_dim).to(device)
        self.target_critic = Critic(state_dim, total_act_dim).to(device)
        
        # Initialize target networks to have the same weights.
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, obs, tau=1.0):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_onehot = self.actor(obs_tensor, tau=tau)
        action = action_onehot.argmax(dim=1).item()
        return action, action_onehot.squeeze(0)
    
    def update(self, samples, all_agents, agent_index):
        states, actions, rewards, next_states, dones = samples
        batch_size = states.shape[0]
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        rewards_agent = torch.FloatTensor(rewards[:, agent_index]).unsqueeze(1).to(self.device)
        
        actions_tensor = torch.FloatTensor(actions.reshape(batch_size, -1)).to(self.device)
        
        # ----- Critic Update -----
        with torch.no_grad():
            next_actions = []
            for agent in all_agents:
                next_act = agent.target_actor(next_states, tau=1.0)
                next_actions.append(next_act)
            next_actions_cat = torch.cat(next_actions, dim=1)
            target_q = self.target_critic(next_states, next_actions_cat)
            y = rewards_agent + self.gamma * target_q * (1 - dones)
        
        current_q = self.critic(states, actions_tensor)
        critic_loss = F.mse_loss(current_q, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ----- Actor Update -----
        actions_split = torch.split(torch.FloatTensor(actions.reshape(batch_size, -1)).to(self.device),
                                    all_agents[0].actor.out.out_features, dim=1)
        local_obs = states  # Using full state as local observation.
        current_action = self.actor(local_obs, tau=1.0)
        actions_split[agent_index] = current_action
        actions_modified = torch.cat(actions_split, dim=1)
        actor_loss = -self.critic(states, actions_modified).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----- Soft Update of Target Networks -----
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        
        return critic_loss.item(), actor_loss.item()
    
    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


# Training Loop

def train_maddpg(env, n_episodes=500, max_steps=200, batch_size=128, buffer_capacity=100000, update_every=100, device='cpu'):
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    # Get an initial observation to determine the flattened state dimension.
    init_obs = env.reset()
    global_state = flatten_obs(init_obs)
    print("Initial flattened state shape:", global_state.shape)  # Debug print
    state_dim = global_state.shape[0]
    
    # For this script, we assume each agent’s observation is the full global state.
    act_dim = 4  # Number of discrete actions.
    total_agents = 1 + env.num_villains  # 1 guardian + villains.
    total_act_dim = total_agents * act_dim  # One-hot vector per agent.
    
    # Create agents: agent 0 is the guardian, agents 1..N are villains.
    agents = []
    agents.append(MADDPGAgent(agent_id=0, obs_dim=state_dim, act_dim=act_dim, 
                              state_dim=state_dim, total_act_dim=total_act_dim, device=device))
    for i in range(1, total_agents):
        agents.append(MADDPGAgent(agent_id=i, obs_dim=state_dim, act_dim=act_dim, 
                                  state_dim=state_dim, total_act_dim=total_act_dim, device=device))
    
    total_steps = 0
    for episode in range(n_episodes):
        obs = env.reset()
        global_state = flatten_obs(obs)
        episode_rewards = np.zeros(total_agents)
        done = False
        step = 0
        
        while not done and step < max_steps:
            actions_list = []
            actions_onehots = []
            # Each agent selects an action based on the current global state.
            for agent in agents:
                a_int, a_onehot = agent.select_action(global_state, tau=1.0)
                actions_list.append(a_int)
                actions_onehots.append(a_onehot.cpu().numpy())
            
            # Build the action dictionary expected by your environment.
            env_actions = {
                "guardian": actions_list[0],
                "villains": np.array(actions_list[1:])
            }
            
            next_obs, rewards_tuple, done, info = env.step(env_actions)
            # For training, assign guardian its own reward and equally split the villain reward.
            rewards = [rewards_tuple[0]] + [rewards_tuple[1] / env.num_villains for _ in range(env.num_villains)]
            next_global_state = flatten_obs(next_obs)
            replay_buffer.add(global_state, np.array(actions_onehots), np.array(rewards), next_global_state, done)
            
            global_state = next_global_state
            episode_rewards += np.array(rewards)
            step += 1
            total_steps += 1
            
            if len(replay_buffer) > batch_size and total_steps % update_every == 0:
                samples = replay_buffer.sample(batch_size)
                for idx, agent in enumerate(agents):
                    c_loss, a_loss = agent.update(samples, agents, agent_index=idx)
                    # Optionally, log or print the losses.
                    
        print(f"Episode {episode+1} Reward: {episode_rewards} Steps: {step} Result: {info.get('result', 'n/a')}")
        
    return agents


# Main Training Entry Point

if __name__ == '__main__':
    # Initialize your environment from env.py.
    env = TreasureGuardianEnv(
        grid_size=10,
        num_villains=3,
        num_keys=5,
        num_pits=3,       # If your env.py still uses "snakes", you can ignore this parameter.
        render_mode=None, # Set to "human" to visualize the environment.
        max_steps=200
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_agents = train_maddpg(env, n_episodes=500, max_steps=200, batch_size=128, device=device)
    
    # Testing the trained agents.
    obs = env.reset()
    global_state = flatten_obs(obs)
    done = False
    while not done:
        actions_list = []
        for agent in trained_agents:
            a_int, _ = agent.select_action(global_state, tau=0.0)  # Greedy action selection.
            actions_list.append(a_int)
        env_actions = {"guardian": actions_list[0], "villains": np.array(actions_list[1:])}
        obs, rewards, done, info = env.step(env_actions)
        global_state = flatten_obs(obs)
        env.render()
    
    env.close()
