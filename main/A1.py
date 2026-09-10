import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")

obs, info = env.reset()

print("\nCHẠY VỚI HÀNH ĐỘNG NGẪU NHIÊN")
for step in range(200):
    action = env.action_space.sample()  # hành động ngẫu nhiên
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Bước {step}: Observation = {obs}, Reward = {reward}")
    
    if terminated or truncated:
        obs, info = env.reset()
        print("Môi trường đã reset")

env.close()