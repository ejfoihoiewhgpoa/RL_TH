import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

env = gym.make("Acrobot-v1", render_mode="rgb_array")

obs, info = env.reset(seed=123)
frames = []

for step in range(200):
    action = env.action_space.sample()  # hành động ngẫu nhiên
    obs, reward, terminated, truncated, info = env.step(action)
    
    frame = env.render()
    frames.append(frame)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()

print(f"Đã thu thập {len(frames)} khung hình.")

# Hiển thị 1 khung hình để kiểm tra
plt.imshow(frames[0])
plt.axis("off")
plt.title("Khung hình đầu tiên - Acrobot")
plt.show()

# Tạo animation từ danh sách frames
fig, ax = plt.subplots()
ax.axis("off")
img = ax.imshow(frames[0])

def update(i):
    img.set_data(frames[i])
    return [img]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)

ani.save("acrobot_animation.gif", writer="pillow", fps=20)
print("Đã lưu animation vào acrobot_animation.gif")

plt.close()