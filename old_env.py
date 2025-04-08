import gymnasium as gym
import numpy as np
import pygame
from collections import defaultdict

class TreasureGuardianEnv(gym.Env):
    def __init__(self, grid_size=10, num_villains=3, num_keys=5, num_pits=3, 
                 max_steps=100, render_mode=None, wall_percentage=0.15):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_villains = num_villains
        self.num_keys = num_keys
        self.num_pits = num_pits         # New parameter for pits
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.wall_percentage = wall_percentage
        
        self.action_space = gym.spaces.Dict({
            "guardian": gym.spaces.Discrete(4),
            "villains": gym.spaces.MultiDiscrete([4] * num_villains)
        })
        
        self.observation_space = gym.spaces.Dict({
            "guardian": gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            "villains": gym.spaces.Box(low=0, high=grid_size-1, shape=(num_villains, 2), dtype=np.int32),
            "keys": gym.spaces.Box(low=0, high=grid_size-1, shape=(num_keys, 2), dtype=np.int32),
            "walls": gym.spaces.Box(low=0, high=grid_size-1, 
                                      shape=(int(grid_size**2 * wall_percentage), 2), dtype=np.int32),
            "treasure": gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            "pits": gym.spaces.Box(low=0, high=grid_size-1, shape=(num_pits, 2), dtype=np.int32)
        })

        self.cell_size = 50
        self.colors = {
            "background": (255, 255, 255),
            "guardian": (0, 0, 255),       # Blue
            "villains": (255, 0, 0),       # Red
            "keys": (255, 215, 0),         # Gold
            "walls": (170, 170, 170),      # Gray
            "treasure": (0, 255, 0),       # Green
            "pits": (0, 0, 0),             # Black
            "grid": (200, 200, 200)
        }
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, 
                                                    self.grid_size * self.cell_size))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Treasure Guardian")
        
        self.reset()
    
    def reset(self):
        self.occupied = set()
        self.walls = self._generate_walls()
        # Generate pits after walls so that keys avoid both.
        self.pits = self._generate_positions(self.num_pits, self.occupied)
        self.guardian_pos = np.array([0, 0], dtype=np.int32)
        self.occupied.add((0, 0))
        self.villains_pos = self._generate_positions(self.num_villains, self.occupied)
        # Track whether each villain holds a key (initially none).
        self.villain_has_key = [False] * len(self.villains_pos)
        self.keys = self._generate_positions(self.num_keys, self.occupied)
        self.treasure = self._generate_positions(1, self.occupied)[0]  # Place treasure
        self.current_step = 0
        self.total_keys_held = 0
        self.keys_collected = set()
        return self._get_obs()
    
    def _generate_walls(self):
        num_walls = int(self.grid_size**2 * self.wall_percentage)
        walls = set()
        while len(walls) < num_walls:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in self.occupied:
                walls.add(pos)
                self.occupied.add(pos)
        return np.array(list(walls), dtype=np.int32)
    
    def _generate_positions(self, count, occupied):
        positions = []
        while len(positions) < count:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in occupied:
                positions.append(pos)
                occupied.add(pos)
        return np.array(positions, dtype=np.int32)
    
    def _respawn_key(self):
        # Generate a valid position for a key (avoid walls and pits).
        valid = False
        while not valid:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos in map(tuple, self.walls) or pos in map(tuple, self.pits):
                continue
            if np.array_equal(np.array(pos, dtype=np.int32), self.guardian_pos):
                continue
            if any(np.array_equal(np.array(pos, dtype=np.int32), villain) for villain in self.villains_pos):
                continue
            valid = True
        return np.array(pos, dtype=np.int32)
    
    def _get_obs(self):
        return {
            "guardian": self.guardian_pos.copy(),
            "villains": self.villains_pos.copy(),
            "keys": self.keys.copy(),
            "walls": self.walls.copy(),
            "treasure": self.treasure.copy(),
            "pits": self.pits.copy()
        }
    
    def _move_agent(self, pos, action):
        # Guardian movement is only blocked by walls.
        x, y = pos
        new_pos = pos.copy()
        if action == 0:  # Up
            new_pos[1] = max(0, y - 1)
        elif action == 1:  # Down
            new_pos[1] = min(self.grid_size - 1, y + 1)
        elif action == 2:  # Left
            new_pos[0] = max(0, x - 1)
        elif action == 3:  # Right
            new_pos[0] = min(self.grid_size - 1, x + 1)
        if tuple(new_pos) in map(tuple, self.walls):
            return pos
        return new_pos
    
    def _move_villain(self, pos, action):
        # Villain movement is blocked by walls.
        x, y = pos
        new_pos = pos.copy()
        if action == 0:
            new_pos[1] = max(0, y - 1)
        elif action == 1:
            new_pos[1] = min(self.grid_size - 1, y + 1)
        elif action == 2:
            new_pos[0] = max(0, x - 1)
        elif action == 3:
            new_pos[0] = min(self.grid_size - 1, x + 1)
        if tuple(new_pos) in map(tuple, self.walls):
            return pos, False  # Movement blocked by wall.
        return new_pos, False
    
    def step(self, actions):
        self.current_step += 1
        guardian_reward = -0.1
        villains_reward = -0.1 * len(self.villains_pos)
        info = {}
        
        # Move guardian.
        self.guardian_pos = self._move_agent(self.guardian_pos, actions["guardian"])
        
        # Check if guardian falls into a pit.
        if any(np.array_equal(self.guardian_pos, pit) for pit in self.pits):
            guardian_reward -= 10
            info["result"] = "villains_win"
            return self._get_obs(), (guardian_reward, villains_reward), True, info
        
        # Check for collisions: if guardian catches any villain.
        remaining_villains = []
        remaining_villain_keys = []
        for i, villain in enumerate(self.villains_pos):
            if np.array_equal(self.guardian_pos, villain):
                guardian_reward += 5      # +5 for catching villain.
                villains_reward -= 5      # Villains penalized.
                # If the caught villain was holding a key, drop it.
                if self.villain_has_key[i]:
                    new_key = self._respawn_key()
                    self.keys = np.concatenate([self.keys, np.array([new_key], dtype=np.int32)], axis=0)
                    self.total_keys_held -= 1
                # Do not add the caught villain.
            else:
                remaining_villains.append(villain)
                remaining_villain_keys.append(self.villain_has_key[i])
        self.villains_pos = np.array(remaining_villains, dtype=np.int32)
        self.villain_has_key = remaining_villain_keys
        
        # Move villains and handle key collection and pit elimination.
        new_villains = []
        new_villain_keys = []
        for i, villain in enumerate(self.villains_pos):
            new_pos, _ = self._move_villain(villain, actions["villains"][i])
            # Check if villain falls into a pit.
            if any(np.array_equal(new_pos, pit) for pit in self.pits):
                villains_reward -= 10  # Penalty for falling in pit.
                continue  # Eliminate villain.
            # Check key collection.
            if (not self.villain_has_key[i]) and any(np.array_equal(new_pos, key) for key in self.keys):
                villains_reward += 5  # Reward for collecting a key.
                new_villain_keys.append(True)
                self.total_keys_held += 1
                # Remove the collected key from the board.
                self.keys = np.array([k for k in self.keys if not np.array_equal(k, new_pos)], dtype=np.int32)
            else:
                new_villain_keys.append(self.villain_has_key[i])
            new_villains.append(new_pos)
        self.villains_pos = np.array(new_villains, dtype=np.int32)
        self.villain_has_key = new_villain_keys
        
        # Terminal conditions:
        # Guardian wins if all villains are eliminated.
        if len(self.villains_pos) == 0:
            guardian_reward += 50
            info["result"] = "guardian_win"
            return self._get_obs(), (guardian_reward, villains_reward), True, info
        # Villains win if they have collected all keys and any villain reaches the treasure.
        if self.total_keys_held == self.num_keys and any(np.array_equal(v, self.treasure) for v in self.villains_pos):
            villains_reward += 10
            info["result"] = "villains_win"
            return self._get_obs(), (guardian_reward, villains_reward), True, info
        # Timeout condition.
        if self.current_step >= self.max_steps:
            info["result"] = "timeout"
            return self._get_obs(), (guardian_reward, villains_reward), True, info
        
        return self._get_obs(), (guardian_reward, villains_reward), False, info
    
    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill(self.colors["background"])
        # Draw walls.
        for x, y in self.walls:
            pygame.draw.rect(self.screen, self.colors["walls"], (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        # Draw pits.
        for x, y in self.pits:
            pygame.draw.rect(self.screen, self.colors["pits"], (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        # Draw keys.
        for x, y in self.keys:
            pygame.draw.rect(self.screen, self.colors["keys"], (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        # Draw treasure.
        pygame.draw.rect(self.screen, self.colors["treasure"], (self.treasure[0] * self.cell_size, self.treasure[1] * self.cell_size, self.cell_size, self.cell_size))
        # Draw guardian.
        pygame.draw.rect(self.screen, self.colors["guardian"], (self.guardian_pos[0] * self.cell_size, self.guardian_pos[1] * self.cell_size, self.cell_size, self.cell_size))
        # Draw villains.
        for x, y in self.villains_pos:
            pygame.draw.rect(self.screen, self.colors["villains"], (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()
        self.clock.tick(10)
    
    def close(self):
        if self.render_mode == "human":
            pygame.quit()

if __name__ == "__main__":
    # Initialize the environment.
    env = TreasureGuardianEnv(
        grid_size=10,
        num_villains=3,
        num_keys=5,
        num_pits=3,
        render_mode="human",
        max_steps=200
    )

    print("Starting example run...")
    
    for episode in range(5):  # Run 2 episodes.
        obs = env.reset()
        done = False
        total_guardian_reward = 0
        total_villain_reward = 0

        while not done:
            # Random actions for guardian and villains.
            actions = {
                "guardian": np.random.randint(4),
                "villains": np.random.randint(4, size=len(obs["villains"]))
            }
            obs, rewards, done, info = env.step(actions)
            total_guardian_reward += rewards[0]
            total_villain_reward += rewards[1]
            env.render()
            
            if done:
                print(f"\nEpisode {episode+1} ended after {env.current_step} steps")
                print(f"Result: {info.get('result', 'unknown')}")
                print(f"Total Guardian Reward: {total_guardian_reward:.2f}")
                print(f"Total Villain Reward: {total_villain_reward:.2f}")
                break
    
    env.close()
    print("\nExample run completed!")



    
