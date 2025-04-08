import gymnasium as gym
import numpy as np
import pygame

class TreasureGuardianEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=10, num_villains=3, num_keys=5, num_pits=3,
                 max_steps=100, render_mode=None, wall_percentage=0.15):
        super().__init__()
        self.grid_size = grid_size
        self.num_villains = num_villains
        self.num_keys = num_keys
        self.num_pits = num_pits
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.wall_percentage = wall_percentage

        # Action and observation spaces
        self.action_space = gym.spaces.Dict({
            "guardian": gym.spaces.Discrete(4),
            "villains": gym.spaces.MultiDiscrete([4] * num_villains)
        })

        num_walls = int(grid_size * grid_size * wall_percentage)
        self.observation_space = gym.spaces.Dict({
            "guardian": gym.spaces.Box(0, grid_size - 1, (2,), dtype=np.int32),
            "villains": gym.spaces.Box(0, grid_size - 1, (num_villains, 2), dtype=np.int32),
            "keys": gym.spaces.Box(0, grid_size - 1, (num_keys, 2), dtype=np.int32),
            "walls": gym.spaces.Box(0, grid_size - 1, (num_walls, 2), dtype=np.int32),
            "treasure": gym.spaces.Box(0, grid_size - 1, (2,), dtype=np.int32),
            "pits": gym.spaces.Box(0, grid_size - 1, (num_pits, 2), dtype=np.int32),
        })

        self.cell_size = 50
        self.colors = {
            "background": (255, 255, 255),
            "guardian": (0, 0, 255),
            "villains": (255, 0, 0),
            "keys": (255, 215, 0),
            "walls": (170, 170, 170),
            "treasure": (0, 255, 0),
            "pits": (0, 0, 0),
        }

        if render_mode == "human":
            pygame.init()
            size = grid_size * self.cell_size
            self.screen = pygame.display.set_mode((size, size))
            pygame.display.set_caption("Treasure Guardian")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.occupied = {(0, 0)}
        self.walls = self._generate_walls()
        self.pits = self._generate_positions(self.num_pits, self.occupied)
        self.guardian_pos = np.array([0, 0], dtype=np.int32)
        self.occupied.add((0, 0))
        self.villains_pos = self._generate_positions(self.num_villains, self.occupied)
        self.villain_has_key = [False] * self.num_villains
        self.keys = self._generate_positions(self.num_keys, self.occupied)
        self.treasure = self._generate_positions(1, self.occupied)[0]
        self.current_step = 0
        self.total_keys_held = 0
        return self._get_obs(), {}

    def _generate_walls(self):
        n = int(self.grid_size * self.grid_size * self.wall_percentage)
        walls = set()
        while len(walls) < n:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in self.occupied:
                walls.add(pos)
                self.occupied.add(pos)
        return np.array(list(walls), dtype=np.int32)

    def _generate_positions(self, count, occupied):
        lst = []
        while len(lst) < count:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in occupied:
                lst.append(pos)
                occupied.add(pos)
        return np.array(lst, dtype=np.int32)

    def _respawn_key(self):
        while True:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if (pos not in map(tuple, self.walls)
                and pos not in map(tuple, self.pits)
                and not np.array_equal(pos, self.guardian_pos)
                and all(not np.array_equal(pos, v) for v in self.villains_pos)):
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
        x, y = pos
        if action == 0: y = max(0, y - 1)
        elif action == 1: y = min(self.grid_size - 1, y + 1)
        elif action == 2: x = max(0, x - 1)
        elif action == 3: x = min(self.grid_size - 1, x + 1)
        new = np.array((x, y), dtype=np.int32)
        return pos if tuple(new) in map(tuple, self.walls) else new

    def step(self, actions):
        self.current_step += 1
        g_r = -0.1
        v_r = -0.1 * len(self.villains_pos)
        info = {}

        self.guardian_pos = self._move_agent(self.guardian_pos, actions["guardian"])

        if any(np.array_equal(self.guardian_pos, p) for p in self.pits):
            g_r -= 10
            info["result"] = "villains_win"
            return self._get_obs(), (g_r, v_r), True, info

        survivors, survivor_keys = [], []
        for i, v in enumerate(self.villains_pos):
            if np.array_equal(self.guardian_pos, v):
                g_r += 5
                v_r -= 5
                if self.villain_has_key[i]:
                    self.total_keys_held -= 1
            else:
                survivors.append(v)
                survivor_keys.append(self.villain_has_key[i])
        self.villains_pos = np.array(survivors, dtype=np.int32)
        self.villain_has_key = survivor_keys

        new_vs, new_vk = [], []
        for i, v in enumerate(self.villains_pos):
            new_pos = self._move_agent(v, actions["villains"][i])
            if any(np.array_equal(new_pos, p) for p in self.pits):
                v_r -= 10
                continue
            if (not self.villain_has_key[i]) and any(np.array_equal(new_pos, k) for k in self.keys):
                v_r += 5
                new_vk.append(True)
                self.total_keys_held += 1
                self.keys = np.array([k for k in self.keys if not np.array_equal(k, new_pos)], dtype=np.int32)
                self.keys = np.vstack([self.keys, self._respawn_key()])
            else:
                new_vk.append(self.villain_has_key[i])
            new_vs.append(new_pos)

        self.villains_pos = np.array(new_vs, dtype=np.int32)
        self.villain_has_key = new_vk

        if len(self.villains_pos) == 0:
            g_r += 50
            info["result"] = "guardian_win"
            return self._get_obs(), (g_r, v_r), True, info

        if self.total_keys_held == self.num_keys and any(np.array_equal(v, self.treasure) for v in self.villains_pos):
            v_r += 10
            info["result"] = "villains_win"
            return self._get_obs(), (g_r, v_r), True, info

        if self.current_step >= self.max_steps:
            info["result"] = "timeout"
            return self._get_obs(), (g_r, v_r), True, info

        return self._get_obs(), (g_r, v_r), False, info

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill(self.colors["background"])
        for x, y in self.walls:
            pygame.draw.rect(self.screen, self.colors["walls"], (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        for x, y in self.pits:
            pygame.draw.rect(self.screen, self.colors["pits"], (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        for x, y in self.keys:
            pygame.draw.rect(self.screen, self.colors["keys"], (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        tx, ty = self.treasure
        pygame.draw.rect(self.screen, self.colors["treasure"], (tx * self.cell_size, ty * self.cell_size, self.cell_size, self.cell_size))
        gx, gy = self.guardian_pos
        pygame.draw.rect(self.screen, self.colors["guardian"], (gx * self.cell_size, gy * self.cell_size, self.cell_size, self.cell_size))
        for x, y in self.villains_pos:
            pygame.draw.rect(self.screen, self.colors["villains"], (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


if __name__ == "__main__":
    env = TreasureGuardianEnv(grid_size=10, num_villains=3, num_keys=5, num_pits=3, render_mode="human", max_steps=200)
    print("Starting example run...")

    for episode in range(5):
        obs, _ = env.reset()
        done = False
        g_total, v_total = 0, 0

        while not done:
            guardian_action = np.random.randint(4)
            villain_action = np.random.randint(4, size=len(obs["villains"])) if len(obs["villains"]) > 0 else np.array([])
            actions = {"guardian": guardian_action, "villains": villain_action}
            obs, (g_r, v_r), done, info = env.step(actions)
            g_total += g_r
            v_total += v_r
            env.render()

        print(f"\nEpisode {episode + 1} ended after {env.current_step} steps")
        print(f"Result: {info.get('result', 'unknown')}")
        print(f"Total Guardian Reward: {g_total:.2f}")
        print(f"Total Villain Reward: {v_total:.2f}")

    env.close()
    print("Example run completed!")
