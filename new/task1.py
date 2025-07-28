class GridWorldEvaluator:
    def __init__(self, rewards, policy, terminal, gamma=1.0):
        self.N = len(rewards)
        self.rewards = rewards
        self.policy = policy
        self.terminal = terminal
        self.gamma = gamma
        self.value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]
        self.value[terminal[0]][terminal[1]] = rewards[terminal[0]][terminal[1]]

    def move(self, action, row, col):
        if action == 'N' and row > 0:
            return row - 1, col
        if action == 'S' and row < self.N - 1:
            return row + 1, col
        if action == 'W' and col > 0:
            return row, col - 1
        if action == 'E' and col < self.N - 1:
            return row, col + 1
        return row, col  # no move if at boundary

    def evaluate_policy(self, theta=1e-4, verbose=True):
        iteration = 0
        while True:
            delta = 0
            for row in range(self.N):
                for col in range(self.N):
                    if (row, col) == self.terminal:
                        continue
                    action = self.policy[row][col]
                    next_row, next_col = self.move(action, row, col)
                    reward = self.rewards[next_row][next_col]
                    new_v = reward + self.gamma * self.value[next_row][next_col]
                    delta = max(delta, abs(self.value[row][col] - new_v))
                    self.value[row][col] = new_v
            iteration += 1
            if verbose:
                print(f"\nIteration {iteration} — Δ = {delta:.6f}")
                for row_vals in self.value:
                    print(['{:.2f}'.format(v) for v in row_vals])
            if delta < theta:
                print(f"\n✅ Converged in {iteration} iterations.")
                break


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

gw = GridWorldEvaluator(rewards=rewards, policy=policy, terminal=(5,5))
gw.evaluate_policy()
