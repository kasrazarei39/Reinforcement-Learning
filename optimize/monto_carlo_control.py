import numpy as np
from collections import defaultdict
import random


class GridWorldMC:
    def __init__(self, rewards, terminal=(5, 5), start=(0, 0), gamma=0.9, epsilon=0.1):
        self.n = rewards.shape[0]
        self.rewards = rewards
        self.terminal = terminal
        self.start = start
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = ['N', 'S', 'E', 'W']

        self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.returns = defaultdict(list)

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

    def generate_episode(self, max_steps=200):
        episode = []
        state = self.start  # configurable starting state
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = self.epsilon_greedy(state)
            next_state, reward, done = self.step(state, action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1
        return episode

    def train(self, episodes=50000):
        for _ in range(episodes):
            episode = self.generate_episode()
            G = 0
            visited = set()
            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                if (state, action) not in visited:
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])
                    visited.add((state, action))

    def extract_policy(self):
        policy = {}
        for state in self.Q:
            best_action = max(self.Q[state], key=self.Q[state].get)
            policy[state] = best_action
        return policy


rewards = np.array([
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -2, -1, -1],
    [-1, -1, -3, -3, -2, -1],
    [-1, -2, -4, -4, -2, -1],
    [-1, -3, -6, -6, -2, -1],
    [-1, -4, -8, -6, -4, 10]
])

env = GridWorldMC(rewards, terminal=(5,5), start=(5,0))
env.train(episodes=5000)
policy = env.extract_policy()

# Print learned policy
for r in range(env.n):
    row = []
    for c in range(env.n):
        if (r,c) == env.terminal:
            row.append("G")
        else:
            row.append(policy.get((r,c), "?"))
    print(row)

