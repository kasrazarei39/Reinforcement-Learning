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

    def generate_episode(self, max_steps=1000, random_start=False):
        episode = []
        if random_start:
            state = (np.random.randint(self.n), np.random.randint(self.n))
        else:
            state = self.start
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = self.epsilon_greedy(state)
            next_state, reward, done = self.step(state, action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1
        return episode, done

    def train(self, episodes=50000, log_interval=5000):
        goal_reached = 0
        returns_log = []

        for ep in range(1, episodes+1):
            episode, reached = self.generate_episode(max_steps=1000, random_start=True)
            G = 0
            visited = set()

            if reached:
                goal_reached += 1

            # Compute returns backward
            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                if (state, action) not in visited:
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])
                    visited.add((state, action))

            # Track episode return
            total_return = sum([r for (_,_,r) in episode])
            returns_log.append(total_return)

            # Decay epsilon
            self.epsilon = max(0.05, self.epsilon * 0.99999)

            # Logging
            if ep % log_interval == 0:
                avg_return = np.mean(returns_log[-log_interval:])
                success_rate = goal_reached / log_interval * 100
                print(f"Episode {ep}: avg_return={avg_return:.2f}, success_rate={success_rate:.1f}%, epsilon={self.epsilon:.3f}")
                goal_reached = 0  # reset counter

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

env = GridWorldMC(rewards, terminal=(5,5), start=(5,0))
env.train(episodes=200000, log_interval=5000)

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
