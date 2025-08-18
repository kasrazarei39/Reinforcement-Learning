import random

class TDnPredictor:
    def __init__(self, rewards, policy, terminal, gamma=1.0, alpha=0.1):
        self.N = len(rewards)
        self.rewards = rewards
        self.policy = policy
        self.terminal = terminal
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
        return row, col  # no move if at edge

    def generate_episode(self, start):
        """Generate a full episode following the policy."""
        episode = []
        row, col = start
        steps = 0
        max_steps = 1000

        while (row, col) != self.terminal and steps < max_steps:
            action = self.policy[row][col]
            next_row, next_col = self.move(action, row, col)
            reward = self.rewards[next_row][next_col]
            episode.append(((row, col), reward))
            row, col = next_row, next_col
            steps += 1

        episode.append(((row, col), 0))  # terminal step
        return episode

    def train(self, n=3, episodes=20, verbose=True):
        for ep in range(1, episodes + 1):
            for row in range(self.N):
                for col in range(self.N):
                    if (row, col) == self.terminal:
                        continue

                    episode = self.generate_episode(start=(row, col))
                    T = len(episode)

                    for t in range(T - n):
                        G = 0
                        for k in range(n):
                            G += (self.gamma ** k) * episode[t + k][1]

                        next_state = episode[t + n][0]
                        G += (self.gamma ** n) * self.value[next_state[0]][next_state[1]]

                        state = episode[t][0]
                        r, c = state
                        self.value[r][c] += self.alpha * (G - self.value[r][c])

            # if verbose:
            #     print(f"\nIteration {ep} — TD({n}) Value Estimates:")
            #     for r in self.value:
            #         print(['{:.2f}'.format(v) for v in r])

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

    td_n = TDnPredictor(rewards, policy, terminal=(5,5), gamma=1.0, alpha=0.1)
    td_n.train(n=1, episodes=5000)

    td_n.print_value()
