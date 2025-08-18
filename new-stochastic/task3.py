class WindyValueIterationSolver:
    def __init__(self, rewards, terminal, gamma=1.0):
        self.N = len(rewards)
        self.rewards = rewards
        self.terminal = terminal
        self.gamma = gamma
        self.value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]
        self.policy = [['' for _ in range(self.N)] for _ in range(self.N)]
        self.actions = ['N', 'S', 'E', 'W']

    def move(self, row, col, d_row, d_col):
        new_row = row + d_row
        new_col = col + d_col
        if 0 <= new_row < self.N and 0 <= new_col < self.N:
            return new_row, new_col
        return row, col

    def get_windy_transitions(self, action, row, col):
        if (row, col) == self.terminal:
            return [(1.0, (row, col))]

        wind_map = {
            'N': [(-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1), (0, 0)],
            'S': [(1, 0), (1, -1), (1, 1), (0, -1), (0, 1), (0, 0)],
            'E': [(0, 1), (-1, 1), (1, 1), (-1, 0), (1, 0), (0, 0)],
            'W': [(0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 0)],
        }

        probs = [0.72, 0.08, 0.08, 0.04, 0.04, 0.04]
        directions = wind_map[action]

        return [(probs[i], self.move(row, col, *directions[i])) for i in range(len(probs))]

    def run_value_iteration(self, theta=1e-4, verbose=True):
        iteration = 0
        while True:
            delta = 0
            new_value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]

            for row in range(self.N):
                for col in range(self.N):
                    if (row, col) == self.terminal:
                        new_value[row][col] = self.rewards[row][col]
                        continue

                    best_value = float('-inf')
                    best_action = None

                    for action in self.actions:
                        expected = 0.0
                        for prob, (next_row, next_col) in self.get_windy_transitions(action, row, col):
                            reward = self.rewards[next_row][next_col]
                            # Absorbing terminal: no γ · V(terminal)
                            if (next_row, next_col) == self.terminal:
                                expected += prob * reward
                            else:
                                expected += prob * (reward + self.gamma * self.value[next_row][next_col])

                        if expected > best_value:
                            best_value = expected
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

solver = WindyValueIterationSolver(rewards=rewards, terminal=(5,5))
solver.run_value_iteration()
