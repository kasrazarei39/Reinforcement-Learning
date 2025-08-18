import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    """
    Custom GridWorld environment following OpenAI Gymnasium API.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=6, rewards=None, start=(5, 0), terminal=(5, 5)):
        super().__init__()

        self.N = grid_size
        self.start = start
        self.terminal = terminal

        # Define action space: 0=N, 1=S, 2=E, 3=W
        self.action_space = spaces.Discrete(4)

        # Observation space is (row, col)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.N),
            spaces.Discrete(self.N)
        ))

        # Rewards grid
        self.rewards = rewards if rewards is not None else np.full((self.N, self.N), -1)
        self.agent_pos = self.start

        # Movement mapping
        self.action_map = {
            0: (-1, 0),  # N
            1: (1, 0),   # S
            2: (0, 1),   # E
            3: (0, -1),  # W
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start
        return self.agent_pos, {}

    def step(self, action):
        if action not in self.action_map:
            raise ValueError("Invalid action")

        # Apply action
        row, col = self.agent_pos
        d_row, d_col = self.action_map[action]
        new_row = np.clip(row + d_row, 0, self.N - 1)
        new_col = np.clip(col + d_col, 0, self.N - 1)
        self.agent_pos = (new_row, new_col)

        # Get reward
        reward = self.rewards[new_row][new_col]

        # Check termination
        terminated = (self.agent_pos == self.terminal)
        truncated = False

        return self.agent_pos, reward, terminated, truncated, {}

    def render(self):
        grid = [[" . " for _ in range(self.N)] for _ in range(self.N)]
        r, c = self.agent_pos
        grid[r][c] = " A "
        grid[self.terminal[0]][self.terminal[1]] = " G "
        print("\n".join("".join(row) for row in grid))
        print()

    def close(self):
        pass



if __name__ == "__main__":
    rewards = np.array([
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -2, -1, -1],
        [-1, -1, -3, -3, -2, -1],
        [-1, -2, -4, -4, -2, -1],
        [-1, -3, -6, -6, -2, -1],
        [-1, -4, -8, -6, -4, 10]
    ])

    env = GridWorldEnv(rewards=rewards)

    obs, info = env.reset()
    env.render()

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, done, trunc, _ = env.step(action)
        print(f"Action: {action} → Obs: {obs}, Reward: {reward}, Done: {done}")
        env.render()
        if done:
            break
