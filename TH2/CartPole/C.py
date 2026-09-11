import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)

manual_actions = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]  # tự chọn tay
total = 0
for a in manual_actions:
    obs, r, terminated, truncated, info = env.step(a)
    total += r
    if terminated or truncated:
        break
print("Manual reward:", total)

import numpy as np

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
                successes.append(truncated)  # sống đủ 500 bước = thành công
                break
        rewards.append(total)
    return np.mean(rewards), np.mean(successes)

r_reward, r_success = run(lambda obs: env.action_space.sample())
print(f"Random -> reward: {r_reward:.2f}, success: {r_success:.2f}")

def heuristic(obs):
    pos, vel, angle, ang_vel = obs
    return 1 if (angle + 0.5 * ang_vel) > 0 else 0  # đẩy theo hướng pole nghiêng

h_reward, h_success = run(heuristic)
print(f"Heuristic -> reward: {h_reward:.2f}, success: {h_success:.2f}")
env.close()

from stable_baselines3 import PPO

model = PPO("MlpPolicy", gym.make("CartPole-v1"), verbose=0)
model.learn(total_timesteps=20_000)