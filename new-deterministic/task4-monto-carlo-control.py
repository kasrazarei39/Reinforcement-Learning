import random
import numpy as np

class MonteCarloControl:
    def __init__(self, rewards, terminal, start, gamma=1.0, epsilon=0.1):
        self.N = len(rewards)
        self.rewards = rewards
        self.terminal = terminal
        self.start = start
        self.gamma = gamma
        self.epsilon = epsilon

        self.actions = ['N', 'S', 'E', 'W']
        self.Q = {
            (row, col): {a: 0.0 for a in self.actions}
            for row in range(self.N) for col in range(self.N)
        }
        self.returns_count = {
            (row, col): {a: 0 for a in self.actions}
            for row in range(self.N) for col in range(self.N)
        }

        self.policy = {
            (row, col): random.choice(self.actions)
            for row in range(self.N) for col in range(self.N)
        }

    def move(self, action, row, col):
        if action == 'N' and row > 0:
            return row - 1, col
        if action == 'S' and row < self.N - 1:
            return row + 1, col
        if action == 'W' and col > 0:
            return row, col - 1
        if action == 'E' and col < self.N - 1:
            return row, col + 1
        return row, col  # no move if out of bounds

    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.Q[state]
        return max(q_values, key=q_values.get)

    def generate_episode(self):
        episode = []
        state = self.start
        steps = 0
        max_steps = 1000

        while state != self.terminal and steps < max_steps:
            action = self.epsilon_greedy_action(state)
            next_state = self.move(action, *state)
            reward = self.rewards[next_state[0]][next_state[1]]
            episode.append((state, action, reward))
            state = next_state
            steps += 1

        return episode
    def update_q_and_policy(self, episodes):
        for ep in range(1, episodes + 1):
            episode = self.generate_episode()
            G = 0
            visited = set()

            for (state, action, reward) in reversed(episode):
                G = self.gamma * G + reward

                if (state, action) not in visited:
                    self.returns_count[state][action] += 1
                    alpha = 1 / self.returns_count[state][action]
                    self.Q[state][action] += alpha * (G - self.Q[state][action])
                    visited.add((state, action))

                    # Improve policy greedily
                    self.policy[state] = max(self.Q[state], key=self.Q[state].get)

    def print_results(self):
        print("\n📌 Learned Policy:")
        for row in range(self.N):
            line = []
            for col in range(self.N):
                if (row, col) == self.terminal:
                    line.append(" G ")
                else:
                    line.append(f" {self.policy[(row, col)][0]} ")
            print("".join(line))

        # print("\n📊 Value Function:")
        # for row in range(self.N):
        #     line = []
        #     for col in range(self.N):
        #         if (row, col) == self.terminal:
        #             line.append("10.00")
        #         else:
        #             best_a = self.policy[(row, col)]
        #             v = self.Q[(row, col)][best_a]
        #             line.append(f"{v:5.2f}")
        #     print(" ".join(line))

if __name__ == "__main__":
    rewards = [
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -2, -1, -1],
        [-1, -1, -3, -3, -2, -1],
        [-1, -2, -4, -4, -2, -1],
        [-1, -3, -6, -6, -2, -1],
        [-1, -4, -8, -6, -4, 10]
    ]

    mc_control = MonteCarloControl(
        rewards=rewards,
        terminal=(5, 5),
        start=(5, 0),
        gamma=1.0,
        epsilon=0.1
    )

    mc_control.update_q_and_policy(episodes=100000)
    mc_control.print_results()
