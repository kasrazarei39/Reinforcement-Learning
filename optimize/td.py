import numpy as np
from collections import defaultdict
import random

class GridWorldTD:
    def __init__(self, rewards, terminal=(5, 5), start=(0, 0), gamma=0.9, epsilon=0.1, alpha=0.1):
        self.n = rewards.shape[0]
        self.rewards = rewards
        self.terminal = terminal
        self.start = start
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.actions = ['N', 'S', 'E', 'W']

        self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def step(self, state, action):
        if state == self.terminal:
            return state, self.rewards[state], True

        r, c = state
        if action == 'N': r = max(r - 1, 0)
        if action == 'S': r = min(r + 1, self.n - 1)
        if action == 'E': c = min(c + 1, self.n - 1)
        if action == 'W': c = max(c - 1, 0)

        next_state = (r, c)
        reward = self.rewards[next_state]
        done = (next_state == self.terminal)
        return next_state, reward, done

    def epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.Q[state]
        return max(q_values, key=q_values.get)

    def train(self, episodes=50000, max_steps=1000, log_interval=5000):
        goal_reached = 0
        returns_log = []

        for ep in range(1, episodes + 1):
            state = self.start
            done = False
            total_return = 0
            steps = 0

            while not done and steps < max_steps:
                action = self.epsilon_greedy(state)
                next_state, reward, done = self.step(state, action)

                # TD(0) update
                best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)
                td_target = reward + self.gamma * self.Q[next_state][best_next_action]
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error

                total_return += reward
                state = next_state
                steps += 1

            if done:
                goal_reached += 1
            returns_log.append(total_return)

            # Decay epsilon
            self.epsilon = max(0.05, self.epsilon * 0.99999)

            # Logging
            if ep % log_interval == 0:
                avg_return = np.mean(returns_log[-log_interval:])
                success_rate = goal_reached / log_interval * 100
                print(f"Episode {ep}: avg_return={avg_return:.2f}, success_rate={success_rate:.1f}%, epsilon={self.epsilon:.3f}")
                goal_reached = 0

    def extract_policy(self):
        policy = {}
        for state in self.Q:
            best_action = max(self.Q[state], key=self.Q[state].get)
            policy[state] = best_action
        return policy

    def extract_values(self):
        V = np.zeros((self.n, self.n))
        for r in range(self.n):
            for c in range(self.n):
                state = (r, c)
                if state == self.terminal:
                    V[r, c] = 0
                elif state in self.Q:
                    V[r, c] = max(self.Q[state].values())
                else:
                    V[r, c] = np.nan
        return V


rewards = np.array([
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -2, -1, -1],
    [-1, -1, -3, -3, -2, -1],
    [-1, -2, -4, -4, -2, -1],
    [-1, -3, -6, -6, -2, -1],
    [-1, -4, -8, -6, -4, 10]
])

env = GridWorldTD(rewards, terminal=(5,5), start=(5,0), alpha=0.1)
env.train(episodes=100000, log_interval=5000)

policy = env.extract_policy()
print("\nFinal policy:")
for r in range(env.n):
    row = []
    for c in range(env.n):
        if (r,c) == env.terminal:
            row.append("G")
        else:
            row.append(policy.get((r,c), "?"))
    print(row)

print("\nValue function:")
print(np.round(env.extract_values(), 1))
