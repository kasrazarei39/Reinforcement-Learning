import numpy as np

class WindyPolicyEvaluator:
    def __init__(self, rewards, policy, terminal, gamma=1.0):
        self.N = len(rewards)
        self.rewards = rewards
        self.policy = policy
        self.terminal = terminal
        self.gamma = gamma
        self.value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]

    def move(self, row, col, d_row, d_col):
        new_row = row + d_row
        new_col = col + d_col
        if 0 <= new_row < self.N and 0 <= new_col < self.N:
            return new_row, new_col
        return row, col  # stay in place if out of bounds

    def get_windy_transitions(self, action, row, col):
        """
        Returns a list of (probability, (next_row, next_col)) tuples
        for all possible stochastic outcomes due to wind.
        """
        if (row, col) == self.terminal:
            return [(1.0, (row, col))]  # ⛔ Absorbing terminal

        if action == 'N':
            return [
                (0.72, self.move(row, col, -1, 0)),   # N
                (0.08, self.move(row, col, -1, -1)),  # NW
                (0.08, self.move(row, col, -1, 1)),   # NE
                (0.04, self.move(row, col, 0, -1)),   # W
                (0.04, self.move(row, col, 0, 1)),    # E
                (0.04, (row, col))                    # stay
            ]
        elif action == 'S':
            return [
                (0.72, self.move(row, col, 1, 0)),    # S
                (0.08, self.move(row, col, 1, -1)),   # SW
                (0.08, self.move(row, col, 1, 1)),    # SE
                (0.04, self.move(row, col, 0, -1)),   # W
                (0.04, self.move(row, col, 0, 1)),    # E
                (0.04, (row, col))
            ]
        elif action == 'E':
            return [
                (0.72, self.move(row, col, 0, 1)),    # E
                (0.08, self.move(row, col, -1, 1)),   # NE
                (0.08, self.move(row, col, 1, 1)),    # SE
                (0.04, self.move(row, col, -1, 0)),   # N
                (0.04, self.move(row, col, 1, 0)),    # S
                (0.04, (row, col))
            ]
        elif action == 'W':
            return [
                (0.72, self.move(row, col, 0, -1)),   # W
                (0.08, self.move(row, col, -1, -1)),  # NW
                (0.08, self.move(row, col, 1, -1)),   # SW
                (0.04, self.move(row, col, -1, 0)),   # N
                (0.04, self.move(row, col, 1, 0)),    # S
                (0.04, (row, col))
            ]
        else:
            return [(1.0, (row, col))]  # Invalid action → no move

    def evaluate_policy(self, threshold=1e-4, verbose=True):
        iteration = 0
        while True:
            delta = 0
            new_value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]
            for row in range(self.N):
                for col in range(self.N):
                    if (row, col) == self.terminal:
                        new_value[row][col] = self.rewards[row][col]
                        continue

                    action = self.policy[row][col]
                    expected_value = 0.0

                    for prob, (next_row, next_col) in self.get_windy_transitions(action, row, col):
                        reward = self.rewards[next_row][next_col]

                        # ✅ If next state is terminal → use only immediate reward
                        if (next_row, next_col) == self.terminal:
                            expected_value += prob * reward
                        else:
                            expected_value += prob * (reward + self.gamma * self.value[next_row][next_col])

                    new_value[row][col] = expected_value
                    delta = max(delta, abs(self.value[row][col] - expected_value))

            self.value = new_value
            iteration += 1

            if verbose:
                print(f"\nIteration {iteration} — Δ = {delta:.6f}")
                for r in self.value:
                    print(['{:.2f}'.format(v) for v in r])

            if delta < threshold:
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

evaluator = WindyPolicyEvaluator(rewards=rewards, policy=policy, terminal=(5,5))
evaluator.evaluate_policy()
