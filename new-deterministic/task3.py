class ValueIterationSolver:
    def __init__(self, rewards, terminal, gamma=1.0):
        self.N = len(rewards)
        self.rewards = rewards
        self.terminal = terminal
        self.gamma = gamma
        self.value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]
        self.policy = [['' for _ in range(self.N)] for _ in range(self.N)]
        self.actions = ['N', 'S', 'E', 'W']

    def move(self, action, row, col):
        if (row, col) == self.terminal:
            return row, col  # 🚫 No transitions from terminal
        if action == 'N' and row > 0:
            return row - 1, col
        elif action == 'S' and row < self.N - 1:
            return row + 1, col
        elif action == 'E' and col < self.N - 1:
            return row, col + 1
        elif action == 'W' and col > 0:
            return row, col - 1
        return row, col  # hit wall

    def run_value_iteration(self, theta=1e-4, verbose=True):
        iteration = 0
        while True:
            delta = 0
            new_value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]
            for row in range(self.N):
                for col in range(self.N):
                    if (row, col) == self.terminal:
                        new_value[row][col] = self.rewards[row][col]  # ✅ Fixed terminal value
                        continue

                    best_value = float('-inf')
                    best_action = None

                    for action in self.actions:
                        next_row, next_col = self.move(action, row, col)
                        reward = self.rewards[next_row][next_col]
                        # ✅ Treat terminal as absorbing — no future value
                        if (next_row, next_col) == self.terminal:
                            val = reward  # Only immediate reward
                        else:
                            val = reward + self.gamma * self.value[next_row][next_col]
                        if val > best_value:
                            best_value = val
                            best_action = action

                    new_value[row][col] = best_value
                    self.policy[row][col] = best_action
                    delta = max(delta, abs(self.value[row][col] - best_value))

            self.value = new_value
            iteration += 1

            if verbose:
                print(f"\nIteration {iteration} — Δ = {delta:.6f}")
                for r in self.value:
                    print(['{:.2f}'.format(v) for v in r])

            if delta < theta:
                print(f"\n✅ Converged in {iteration} iterations.")
                break

        print("\n🧭 Optimal Policy:")
        for row in self.policy:
            print(row)



rewards = [
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -2, -1, -1],
    [-1, -1, -3, -3, -2, -1],
    [-1, -2, -4, -4, -2, -1],
    [-1, -3, -6, -6, -2, -1],
    [-1, -4, -8, -6, -4, 10]
]

solver = ValueIterationSolver(rewards, terminal=(5,5))
solver.run_value_iteration()