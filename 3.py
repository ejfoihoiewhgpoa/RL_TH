import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")

print(env.action_space)
print(env.observation_space)

obs, info = env.reset()

for step in range(500):
    action = env.action_space.sample()  # hành động ngẫu nhiên
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Bước {step}: Observation = {obs}, Reward = {reward}")
    
    if terminated or truncated:
        obs, info = env.reset()
        print("Môi trường đã reset")

env.close()