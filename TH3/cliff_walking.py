import gymnasium as gym
import numpy as np

def value_iteration(env, gamma=0.9, theta=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    P = env.unwrapped.P                  # lấy bảng chuyển trạng thái 1 lần ở đầu
    V = np.zeros(n_states)               # khởi tạo V = 0

    def q_values(s):                     # giá trị của các action tại s
        q = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, ss, r, done in P[s][a]:
                q[a] += prob * (r + gamma * V[ss] * (not done))   # (not done): hết episode thì không có tương lai
        return q

    while True:
        delta = 0.0
        for s in range(n_states):
            best = q_values(s).max()
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        # delta là thay đổi lớn nhất của V trong một sweep. Phép cập nhật là
        # phép co hệ số gamma, nên delta < theta nghĩa là V gần V* (sai số cỡ theta*gamma/(1-gamma)).
        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        policy[s] = q_values(s).argmax()
    return V, policy


env = gym.make("CliffWalking-v1")
n_states = env.observation_space.n
n_actions = env.action_space.n
print("States:", n_states, "Actions:", n_actions)

V, policy = value_iteration(env)

arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
for row in range(4):
    print(" ".join(arrows[policy[row * 12 + col]] for col in range(12)))

state, _ = env.reset()
total, done = 0, False
while not done:
    state, r, terminated, truncated, _ = env.step(policy[state])
    total += r
    done = terminated or truncated
print("Total reward:", total)
print(V.reshape(4, 12).round(2))