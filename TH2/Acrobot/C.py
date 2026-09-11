import gymnasium as gym
import numpy as np 

env2 = gym.make("Acrobot-v1")
obs, info = env2.reset(seed=0)

manual_actions = [2, 0, 2, 0, 2, 2, 0, 0, 2, 2]  # tự chọn tay
total = 0
for a in manual_actions:
    obs, r, terminated, truncated, info = env2.step(a)
    total += r
    if terminated or truncated:
        break
print("Manual reward:", total)

def run2(policy_fn, n=50):
    rewards, successes = [], []
    for i in range(n):
        obs, info = env2.reset(seed=i)
        total = 0
        for step in range(500):
            action = policy_fn(obs)
            obs, r, terminated, truncated, info = env2.step(action)
            total += r
            if terminated or truncated:
                successes.append(terminated)  # đạt goal trước timeout = thành công
                break
        rewards.append(total)
    return np.mean(rewards), np.mean(successes)

r_reward2, r_success2 = run2(lambda obs: env2.action_space.sample())
print(f"Random -> reward: {r_reward2:.2f}, success: {r_success2:.2f}")

def heuristic2(obs):
    _, _, _, _, vel1, vel2 = obs
    return 2 if vel2 > 0 else (0 if vel2 < 0 else 1)  # đánh đu theo hướng vận tốc

h_reward2, h_success2 = run2(heuristic2)
print(f"Heuristic -> reward: {h_reward2:.2f}, success: {h_success2:.2f}")
env2.close()

from stable_baselines3 import PPO

model2 = PPO("MlpPolicy", gym.make("Acrobot-v1"), verbose=0)
model2.learn(total_timesteps=20_000)