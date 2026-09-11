import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")

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
                successes.append(truncated)
                break
        rewards.append(total)
    return np.mean(rewards), np.mean(successes)

# Random
r_reward, r_success = run(lambda obs: env.action_space.sample())

# Heuristic
def heuristic(obs):
    pos, vel, angle, ang_vel = obs
    return 1 if (angle + 0.5 * ang_vel) > 0 else 0

h_reward, h_success = run(heuristic)

print(f"Random    -> reward: {r_reward:.2f}, success: {r_success:.2f}")
print(f"Heuristic -> reward: {h_reward:.2f}, success: {h_success:.2f}")
env.close()