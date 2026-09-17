import gymnasium as gym
import pygame

# Tạo môi trường với chế độ hiển thị cửa sổ
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

print("--- ĐIỀU KHIỂN CARTPOLE ---")
print("Phím Mũi tên Trái (<-): Đẩy xe sang trái")
print("Phím Mũi tên Phải (->): Đẩy xe sang phải")
print("Bấm ESC để thoát.")

running = True
action = 0  # Mặc định 0 là sang trái, 1 là sang phải

while running:
    # Lấy sự kiện từ bàn phím
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 0
            elif event.key == pygame.K_RIGHT:
                action = 1
            elif event.key == pygame.K_ESCAPE:
                running = False

    # Thực hiện hành động trong môi trường
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Gậy đã ngã! Đang reset lại game...")
        observation, info = env.reset()

env.close()