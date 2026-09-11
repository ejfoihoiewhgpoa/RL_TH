import gymnasium as gym
import numpy as np

env = gym.make("Acrobot-v1")

def run(policy_fn, n=50):
    rewards, successes = [], []
    for i in range(n):
        obs, info = env.reset(seed=i)
        total = 0
        for step in range(500):
            action = policy_fn(obs)
            obs, r, terminated, truncated, info = env.step(action)
            total += r
            if terminated or truncated:
                successes.append(terminated)
                break
        rewards.append(total)
    return np.mean(rewards), np.mean(successes)

# Random
r_reward, r_success = run(lambda obs: env.action_space.sample())

# Heuristic
def heuristic(obs):
    _, _, _, _, vel1, vel2 = obs
    return 2 if vel2 > 0 else (0 if vel2 < 0 else 1)

h_reward, h_success = run(heuristic)

print(f"Random    -> reward: {r_reward:.2f}, success: {r_success:.2f}")
print(f"Heuristic -> reward: {h_reward:.2f}, success: {h_success:.2f}")
env.close()