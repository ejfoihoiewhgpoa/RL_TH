import gymnasium as gym
import numpy as np

def run_episode_random(env, seed):
    obs, info = env.reset(seed=seed)
    total_reward = 0
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return total_reward, step + 1
    return total_reward, 500

env = gym.make("CartPole-v1")
u = env.unwrapped

# Thay đổi tham số
u.length = 1.5              # pole dài hơn (mặc định 0.5)
u.polemass_length = u.masspole * u.length
u.gravity = 9.8              # trọng lực (mặc định 9.8)
u.force_mag = 10.0           # lực đẩy (mặc định 10.0)

rewards = [run_episode_random(env, seed=i)[0] for i in range(50)]
print(f"Avg reward: {np.mean(rewards):.2f}")
env.close()