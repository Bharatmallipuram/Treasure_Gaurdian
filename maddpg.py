
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import time
import random
from collections import deque
from env import TreasureGuardianEnv, LightTreasureGuardianEnv

# MADDPG Implementation
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(Critic, self).__init__()
        self.total_actions = num_agents * action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + self.total_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, actions):
        return self.net(torch.cat([state, actions], dim=1))

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, num_agents, agent_idx):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(state_dim, action_dim, num_agents)
        self.critic_target = Critic(state_dim, action_dim, num_agents)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.001)

        self.action_dim = action_dim
        self.agent_idx = agent_idx
        self.num_agents = num_agents

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            probs = self.actor(state)
        return torch.argmax(probs).item()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, actions, rewards, next_state, done):
        self.buffer.append((state, actions, rewards, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)


# Helper Functions and Training Parameters
def pad_array(arr, target_len):
    flat = arr.flatten()
    if len(flat) < target_len:
        flat = np.concatenate([flat, np.zeros(target_len - len(flat))])
    return flat

GRID_SIZE = 10
NUM_VILLAINS = 1
NUM_KEYS = 1
NUM_PITS = 2
WALL_PERCENT = 0.15
NUM_WALLS = int(GRID_SIZE * GRID_SIZE * WALL_PERCENT)
STATE_DIM = 2 + (NUM_VILLAINS * 2) + (NUM_KEYS * 2) + (NUM_WALLS * 2) + 2 + (NUM_PITS * 2)
ACTION_DIM = 4
NUM_AGENTS = 1 + NUM_VILLAINS

def flatten_observation(obs):
    return np.concatenate([
        pad_array(obs['guardian'], 2),
        pad_array(obs['villains'], NUM_VILLAINS * 2),
        pad_array(obs['keys'], NUM_KEYS * 2),
        pad_array(obs['walls'], NUM_WALLS * 2),
        pad_array(obs['treasure'], 2),
        pad_array(obs['pits'], NUM_PITS * 2)
    ])

BATCH_SIZE = 128
BUFFER_CAPACITY = 100000
NUM_EPISODES = 50
MAX_STEPS = 20
GAMMA = 0.99
TAU = 0.01
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

# Main Function: Training with Simulation during Episodes
# If True, the training loop will render the environment at each step.
SIMULATE_DURING_TRAINING = True

def main():
    # Create the training environment with rendering enabled for simulation.
    env = TreasureGuardianEnv(
        grid_size=GRID_SIZE,
        num_villains=NUM_VILLAINS,
        num_keys=NUM_KEYS,
        num_pits=NUM_PITS,
        wall_percentage=WALL_PERCENT,
        render_mode="human" if SIMULATE_DURING_TRAINING else None,
        max_steps=MAX_STEPS
    )

    agents = [MADDPGAgent(STATE_DIM, ACTION_DIM, NUM_AGENTS, i) for i in range(NUM_AGENTS)]
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    epsilon = EPS_START

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        state = flatten_observation(obs)
        episode_rewards = [0] * NUM_AGENTS

        for step in range(MAX_STEPS):
            actions = [agent.act(state, epsilon) for agent in agents]
            env_action = {
                "guardian": actions[0],
                "villains": np.array(actions[1:])
            }
            next_obs, (g_reward, v_reward), done, _ = env.step(env_action)
            next_state = flatten_observation(next_obs)
            rewards = [g_reward] + [v_reward] * NUM_VILLAINS
            buffer.push(state, actions, rewards, next_state, done)
            state = next_state
            for i in range(NUM_AGENTS):
                episode_rewards[i] += rewards[i]

            # Render simulation during training (if flag is True).
            if SIMULATE_DURING_TRAINING:
                env.render()
                # You may adjust the sleep delay if needed.
                time.sleep(0.05)

            if len(buffer) >= BATCH_SIZE:
                states, actions_sample, rewards_sample, next_states, dones = buffer.sample(BATCH_SIZE)
                states = torch.FloatTensor(states)
                actions_sample = torch.LongTensor(actions_sample)
                rewards_sample = torch.FloatTensor(rewards_sample)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                for agent in agents:
                    actions_onehot = torch.zeros(BATCH_SIZE, NUM_AGENTS * ACTION_DIM)
                    for i in range(NUM_AGENTS):
                        actions_onehot[:, i * ACTION_DIM:(i+1)*ACTION_DIM] = F.one_hot(
                            actions_sample[:, i], num_classes=ACTION_DIM).float()

                    with torch.no_grad():
                        target_actions = []
                        for a in agents:
                            target_probs = a.actor_target(next_states)
                            target_acts = F.one_hot(torch.argmax(target_probs, dim=1),
                                                    num_classes=ACTION_DIM).float()
                            target_actions.append(target_acts)
                        target_actions = torch.cat(target_actions, dim=1)
                        q_next = agent.critic_target(next_states, target_actions)
                        q_target = rewards_sample[:, agent.agent_idx] + GAMMA * (1 - dones) * q_next.squeeze()

                    current_q = agent.critic(states, actions_onehot)
                    critic_loss = F.mse_loss(current_q.squeeze(), q_target)
                    agent.critic_optim.zero_grad()
                    critic_loss.backward()
                    agent.critic_optim.step()

                    probs = agent.actor(states)
                    current_acts = F.one_hot(torch.argmax(probs, dim=1), num_classes=ACTION_DIM).float()
                    new_actions = actions_onehot.clone()
                    new_actions[:, agent.agent_idx * ACTION_DIM:(agent.agent_idx+1)*ACTION_DIM] = current_acts
                    actor_loss = -agent.critic(states, new_actions).mean()
                    agent.actor_optim.zero_grad()
                    actor_loss.backward()
                    agent.actor_optim.step()

                    for param, target_param in zip(agent.actor.parameters(), agent.actor_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                    for param, target_param in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        print(f"Episode {episode+1}/{NUM_EPISODES}")
        print(f"Guardian Reward: {episode_rewards[0]:.2f}")
        print(f"Villains Reward: {sum(episode_rewards[1:]):.2f}")
        print(f"Epsilon: {epsilon:.3f}\n")

    for i, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(), f"agent_{i}_actor.pth")
        torch.save(agent.critic.state_dict(), f"agent_{i}_critic.pth")
    env.close()

if __name__ == "__main__":
    main()
