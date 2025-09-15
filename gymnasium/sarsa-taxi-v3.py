import gymnasium as gym
import numpy as np
import random
from collections import defaultdict

class SARSALearner:
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_actions = env.action_space.n

        # Q-table
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))

    def epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.Q[state]))

    def train(self, episodes=10000, max_steps=200, log_interval=1000):
        rewards_per_ep = []

        for ep in range(1, episodes + 1):
            state, _ = self.env.reset()
            action = self.epsilon_greedy(state)
            total_reward = 0

            for t in range(max_steps):
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if not done:
                    next_action = self.epsilon_greedy(next_state)
                    td_target = reward + self.gamma * self.Q[next_state][next_action]
                else:
                    td_target = reward

                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error

                total_reward += reward
                state, action = next_state, (next_action if not done else None)

                if done:
                    break

            rewards_per_ep.append(total_reward)
            self.epsilon = max(0.05, self.epsilon * 0.999)  # decay

            if ep % log_interval == 0:
                avg = np.mean(rewards_per_ep[-log_interval:])
                print(f"Episode {ep}: avg_reward={avg:.3f}, epsilon={self.epsilon:.3f}")

        return rewards_per_ep

    def extract_policy(self):
        return {s: int(np.argmax(self.Q[s])) for s in self.Q}


# --- Run with FrozenLake ---
env = gym.make("Taxi-v3")
agent = SARSALearner(env, alpha=0.1, epsilon=0.2)

agent.train(episodes=5000, log_interval=500)

print("\nLearned policy:")
print(agent.extract_policy())
