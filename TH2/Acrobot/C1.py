import gymnasium as gym
import pygame

env = gym.make("Acrobot-v1", render_mode="human")
observation, info = env.reset()

print("--- ĐIỀU KHIỂN ACROBOT ---")
print("Phím 'A': Mô-men xoắn ngược chiều kim đồng hồ (Action 0)")
print("Phím 'S': Không tác dụng lực (Action 1)")
print("Phím 'D': Mô-men xoắn thuận chiều kim đồng hồ (Action 2)")
print("Bấm ESC để thoát.")

running = True
action = 1  # Mặc định 1 là không tác dụng lực

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                action = 0
            elif event.key == pygame.K_s:
                action = 1
            elif event.key == pygame.K_d:
                action = 2
            elif event.key == pygame.K_ESCAPE:
                running = False

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Chúc mừng! Bạn đã văng chạm vạch đích (hoặc hết giờ)!")
        observation, info = env.reset()

env.close()