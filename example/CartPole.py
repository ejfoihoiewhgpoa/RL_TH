#CartPole-v1
import gymnasium as gym

# Tạo môi trường
env = gym.make("CartPole-v1", render_mode="human")

# Đặt lại môi trường
obs, info = env.reset()

for _ in range(500):
    action = env.action_space.sample() # hành động ngẫu nhiên
    obs, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        obs, info = env.reset()

env.close()