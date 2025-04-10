
import gymnasium as gym
import pygame
import numpy as np


# Environment Definitions
class TreasureGuardianEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=10, num_villains=1, num_keys=3, num_pits=2,
                 max_steps=100, render_mode=None, wall_percentage=0.15, colors=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_villains = num_villains
        self.num_keys = num_keys
        self.num_pits = num_pits
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.wall_percentage = wall_percentage
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
        # Default colors; note that a "background" color is needed for rendering.
        self.colors = colors or {
            "background": (0, 0, 0),         # Black
            "guardian": (0, 255, 255),       # Electric Cyan
            "villains": (255, 0, 255),       # Hot Magenta
            "keys": (57, 255, 20),           # Neon Green
            "treasure": (255, 191, 0),       # Glowing Amber
            "walls": (47, 79, 79),           # Graphite Gray
            "pits": (139, 0, 0),             # Deep Red
            "grid": (112, 128, 144),
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
        self.villain_keys = [0] * self.num_villains
        self.keys = self._generate_positions(self.num_keys, self.occupied)
        self.treasure = self._generate_positions(1, self.occupied)[0]
        self.current_step = 0
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
        if action == 0:   # up
            y = max(0, y - 1)
        elif action == 1: # down
            y = min(self.grid_size - 1, y + 1)
        elif action == 2: # left
            x = max(0, x - 1)
        elif action == 3: # right
            x = min(self.grid_size - 1, x + 1)
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
            info["result"] = "villain_win"
            return self._get_obs(), (g_r, v_r), True, info

        updated_villains = []
        updated_keys = self.keys.tolist()
        updated_villain_keys = []
        for i, v in enumerate(self.villains_pos):
            new_v = self._move_agent(v, actions["villains"][i])
            if any(np.array_equal(new_v, p) for p in self.pits):
                v_r -= 10
                continue
            key_collected = False
            for key in self.keys:
                if np.array_equal(new_v, key):
                    key_collected = True
                    v_r += 5
                    updated_villain_keys.append(self.villain_keys[i] + 1)
                    updated_keys.remove(key.tolist())
                    break
            if not key_collected:
                updated_villain_keys.append(self.villain_keys[i])
            updated_villains.append(new_v)
        self.villains_pos = np.array(updated_villains, dtype=np.int32)
        self.villain_keys = updated_villain_keys
        self.keys = np.array(updated_keys, dtype=np.int32)

        survivors = []
        survivors_keys = []
        for i, v in enumerate(self.villains_pos):
            if np.array_equal(self.guardian_pos, v):
                g_r += 5
                v_r -= 5
            else:
                survivors.append(v)
                survivors_keys.append(self.villain_keys[i])
        self.villains_pos = np.array(survivors, dtype=np.int32)
        self.villain_keys = survivors_keys

        if len(self.villains_pos) == 0:
            info["result"] = "guardian_win"
            g_r += 50
            return self._get_obs(), (g_r, v_r), True, info
        for i, v in enumerate(self.villains_pos):
            if np.array_equal(v, self.treasure) and self.villain_keys[i] == self.num_keys:
                info["result"] = "villain_win"
                v_r += 50
                return self._get_obs(), (g_r, v_r), True, info
        if self.current_step >= self.max_steps:
            info["result"] = "timeout"
            return self._get_obs(), (g_r, v_r), True, info

        return self._get_obs(), (g_r, v_r), False, info

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill(self.colors["background"])
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(
                    self.screen,
                    self.colors["grid"],
                    (i * self.cell_size, j * self.cell_size, self.cell_size, self.cell_size),
                    width=1
                )
        for pos in self.walls:
            x, y = pos
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors["walls"], rect)
        def draw_entity(pos, color, radius_ratio=0.35):
            x, y = pos
            center = (x * self.cell_size + self.cell_size // 2,
                      y * self.cell_size + self.cell_size // 2)
            radius = int(self.cell_size * radius_ratio)
            pygame.draw.circle(self.screen, color, center, radius)
        for pos in self.pits:
            draw_entity(pos, self.colors["pits"])
        for pos in self.keys:
            draw_entity(pos, self.colors["keys"])
        draw_entity(self.treasure, self.colors["treasure"])
        draw_entity(self.guardian_pos, self.colors["guardian"])
        for pos in self.villains_pos:
            draw_entity(pos, self.colors["villains"])
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


class LightTreasureGuardianEnv(TreasureGuardianEnv):
    def __init__(self, **kwargs):
        colors = {
            "background": (10, 10, 30),         # Deep Space Blue
            "guardian": (0, 255, 255),          # Neon Cyan (Cyber Guardian)
            "villains": (255, 0, 255),          # Magenta (Glitchy Invaders)
            "keys": (255, 255, 0),              # Bright Yellow (Power Modules)
            "walls": (120, 120, 120),           # Metallic Grey (Steel Barriers)
            "treasure": (0, 255, 127),          # Neon Green (Energy Vault)
            "pits": (255, 0, 0),                # Danger Red (Lava Traps)
            "grid": (70, 130, 180),             # Steel Blue (HUD Grid)
        }
        super().__init__(colors=colors, **kwargs)
