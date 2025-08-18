import random

class TDPredictor:
    def __init__(self, rewards, policy, terminal, start, gamma=1.0, alpha=0.1):
        self.N = len(rewards)
        self.rewards = rewards
        self.policy = policy
        self.terminal = terminal
        self.start = start
        self.gamma = gamma
        self.alpha = alpha

        self.value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]

    def move(self, action, row, col):
        if action == 'N' and row > 0:
            return row - 1, col
        if action == 'S' and row < self.N - 1:
            return row + 1, col
        if action == 'W' and col > 0:
            return row, col - 1
        if action == 'E' and col < self.N - 1:
            return row, col + 1
        return row, col

    def run_episode(self, start=None):
        if start is None:
            row, col = self.start
        else:
            row, col = start
        steps = 0
        max_steps = 1000

        while (row, col) != self.terminal and steps < max_steps:
            action = self.policy[row][col]
            next_row, next_col = self.move(action, row, col)
            reward = self.rewards[next_row][next_col]

            # TD(0) update
            v_current = self.value[row][col]
            v_next = self.value[next_row][next_col]
            td_target = reward + self.gamma * v_next
            td_error = td_target - v_current
            self.value[row][col] += self.alpha * td_error

            row, col = next_row, next_col
            steps += 1

    def train(self, episodes=10, verbose=False):
        for ep in range(1, episodes + 1):
            for row in range(self.N):
                for col in range(self.N):
                    if (row, col) != self.terminal:
                        self.run_episode(start=(row, col))
            if verbose:
                print(f"Iteration {ep} complete.")

    def print_value(self):
        print("\n📊 State Value Function (TD Prediction):")
        for r in self.value:
            print(['{:.2f}'.format(v) for v in r])

if __name__ == "__main__":
    rewards = [
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -2, -1, -1],
        [-1, -1, -3, -3, -2, -1],
        [-1, -2, -4, -4, -2, -1],
        [-1, -3, -6, -6, -2, -1],
        [-1, -4, -8, -6, -4, 10]
    ]

    policy = [
        ['E','E','E','E','E','S'],
        ['E','E','E','S','E','S'],
        ['E','E','S','S','S','S'],
        ['E','S','S','S','S','S'],
        ['E','S','S','S','S','S'],
        ['E','E','E','E','E','E']
    ]

    td = TDPredictor(
        rewards=rewards,
        policy=policy,
        terminal=(5, 5),
        start=(5, 0),
        gamma=1.0,
        alpha=0.1
    )

    td.train(episodes=5000)
    td.print_value()
