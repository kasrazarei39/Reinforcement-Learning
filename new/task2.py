class MonteCarloEvaluator:
    def __init__(self, rewards, policy, terminal, start, gamma=1.0):
        self.N = len(rewards)
        self.rewards = rewards
        self.policy = policy
        self.terminal = terminal
        self.start = start
        self.gamma = gamma
        self.value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]
        self.counts = [[0 for _ in range(self.N)] for _ in range(self.N)]

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

    def generate_episode(self, start):
        episode = []
        row, col = start

        while (row, col) != self.terminal:
            action = self.policy[row][col]
            next_row, next_col = self.move(action, row, col)
            reward = self.rewards[next_row][next_col]
            episode.append(((row, col), reward))
            row, col = next_row, next_col

        episode.append((self.terminal, self.rewards[self.terminal[0]][self.terminal[1]]))
        return episode

    def evaluate_policy(self, episodes=500, verbose=False):
        for ep in range(episodes):
            for row in range(self.N):
                for col in range(self.N):
                    if (row, col) == self.terminal:
                        continue
                    episode = self.generate_episode(start=(row, col))
                    G = 0
                    for (state, reward) in reversed(episode):
                        r, c = state
                        G = self.gamma * G + reward
                        self.counts[r][c] += 1
                        alpha = 1 / self.counts[r][c]
                        self.value[r][c] += alpha * (G - self.value[r][c])

        if verbose:
            print("\n📊 Final Value Estimates (Monte Carlo):")
            for r in self.value:
                print(['{:.2f}'.format(v) for v in r])



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

mc = MonteCarloEvaluator(rewards=rewards, policy=policy, terminal=(5,5), start=(5,0))
mc.evaluate_policy(episodes=500, verbose=True)
