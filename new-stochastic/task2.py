import random

class WindyMonteCarloEvaluator:
    def __init__(self, rewards, policy, terminal, start, gamma=1.0):
        self.N = len(rewards)
        self.rewards = rewards
        self.policy = policy
        self.terminal = terminal
        self.start = start
        self.gamma = gamma
        self.value = [[0.0 for _ in range(self.N)] for _ in range(self.N)]
        self.counts = [[0 for _ in range(self.N)] for _ in range(self.N)]

    def move(self, row, col, d_row, d_col):
        new_row = row + d_row
        new_col = col + d_col
        if 0 <= new_row < self.N and 0 <= new_col < self.N:
            return new_row, new_col
        return row, col

    def sample_windy_action(self, action, row, col):
        """
        Sample a stochastic outcome given an action.
        """
        if (row, col) == self.terminal:
            return row, col

        # Probabilistic transitions
        choices = {
            'N': [
                (0.72, (-1, 0)), (0.08, (-1, -1)), (0.08, (-1, 1)),
                (0.04, (0, -1)), (0.04, (0, 1)), (0.04, (0, 0))
            ],
            'S': [
                (0.72, (1, 0)), (0.08, (1, -1)), (0.08, (1, 1)),
                (0.04, (0, -1)), (0.04, (0, 1)), (0.04, (0, 0))
            ],
            'E': [
                (0.72, (0, 1)), (0.08, (-1, 1)), (0.08, (1, 1)),
                (0.04, (-1, 0)), (0.04, (1, 0)), (0.04, (0, 0))
            ],
            'W': [
                (0.72, (0, -1)), (0.08, (-1, -1)), (0.08, (1, -1)),
                (0.04, (-1, 0)), (0.04, (1, 0)), (0.04, (0, 0))
            ]
        }

        rand = random.random()
        cum_prob = 0.0
        for prob, (d_row, d_col) in choices[action]:
            cum_prob += prob
            if rand < cum_prob:
                return self.move(row, col, d_row, d_col)
        return row, col

    def generate_episode(self, start):
        """
        Simulates an episode from a given start, using the policy.
        Returns a list of ((row, col), reward) pairs.
        """
        episode = []
        row, col = start

        while (row, col) != self.terminal:
            action = self.policy[row][col]
            next_row, next_col = self.sample_windy_action(action, row, col)
            reward = self.rewards[next_row][next_col]
            episode.append(((row, col), reward))
            row, col = next_row, next_col

        return episode

    def evaluate_policy(self, episodes=1000, verbose=True):
        for ep in range(episodes):
            for row in range(self.N):
                for col in range(self.N):
                    if (row, col) == self.terminal:
                        continue

                    episode = self.generate_episode((row, col))
                    G = 0
                    for (state, reward) in reversed(episode):
                        r, c = state
                        G = self.gamma * G + reward
                        self.counts[r][c] += 1
                        alpha = 1 / self.counts[r][c]
                        self.value[r][c] += alpha * (G - self.value[r][c])

        if verbose:
            print("\n📊 Final Value Estimates (Monte Carlo with Wind):")
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

mc = WindyMonteCarloEvaluator(
    rewards=rewards,
    policy=policy,
    terminal=(5, 5),
    start=(5, 0),
    gamma=1.0
)

mc.evaluate_policy(episodes=1000)
