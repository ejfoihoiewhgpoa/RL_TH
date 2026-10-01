import gymnasium as gym
import numpy as np

def value_iteration(env, gamma=0.9, theta=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states) # khởi tạo cho các giá trị ban đầu bằng 0
    
    def q_values(s): # tính giá trị của các action tại s
        q = np.zeros(n_actions)
        for a in range(n_actions):
                for prob, ss, r, done in P[s][a]:
                    q[a] += prob * (r + gamma * V[ss])
        return q

    while True:
        P = env.unwrapped.P  # lấy bảng chuyển trạng thái từ môi trường gốc
        delta = 0
        for s in range(n_states):
            best = q_values(s).max() # lấy giá trị tốt nhất của q_values
            delta = max(delta, abs(best - V[s])) # cho thấy sự khác nhau giữa giá trị lớn nhất và giá trị hiện tại
            V[s] = best
            # delta là thay đổi lớn nhất của V(s) trong một lần quét toàn bộ state.
            # Vì Value Iteration là phép co với hệ số gamma, nếu delta < theta
            # thì V đã gần như không đổi, sai số so với V* nhỏ (cỡ theta*gamma/(1-gamma)).
            # Khi đó ta dừng lại.
        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        policy[s] = q_values(s).argmax()

    return V, policy

def policy_iteration(env, gamma=0.9, theta=1e-6):
    """
    Policy Iteration cho môi trường Taxi-v3.
    """
    n_states = env.observation_space.n      # 500
    n_actions = env.action_space.n          # 6
    P = env.unwrapped.P                     # P[s][a] -> [(prob, next_s, reward, done)]

    # 1. Tại sao tách riêng Policy Evaluation và Policy Improvement?
    #    Evaluation trả lời câu hỏi "chính sách NÀY tốt đến đâu?" (tính V^pi).
    #    Improvement trả lời câu hỏi "với các giá trị đó, có làm tốt hơn được
    #    không?" (chọn tham lam theo V^pi). Theo định lý cải thiện chính sách
    #    (policy improvement theorem), hành động tham lam theo V^pi cho ra một
    #    chính sách không tệ hơn pi, nên mỗi vòng không bao giờ xấu đi. Việc tách
    #    riêng cũng giúp ta kiểm tra được chính sách đã ổn định hay chưa.
    #
    # 2. Tại sao evaluation phải lặp (iterative) dù chính sách đã cố định?
    #    Với pi cố định, V^pi(s) = sum P * [R + gamma * V^pi(s')] là một hệ
    #    phương trình tuyến tính: mỗi V(s) phụ thuộc vào V của các state khác
    #    (đôi khi cả chính nó, ví dụ khi đâm vào tường). Một lượt quét không thể
    #    cho giá trị chính xác vì V(s') chưa biết. Các lượt quét lặp lại tạo
    #    thành một phép co với hệ số gamma và hội tụ về điểm bất động duy nhất.
    #    (Giải trực tiếp hệ tuyến tính cũng được, nhưng tốn O(n^3).)
    #
    # 3. Khi nào chính sách ngừng thay đổi?
    #    Khi chính sách tham lam theo V^pi trùng với chính pi. Lúc đó pi thỏa
    #    phương trình Bellman optimality, nên pi là chính sách tối ưu.
    # --------------------------------------------------------------------------------

    policy = np.random.choice(n_actions, n_states)   # 1. chính sách ngẫu nhiên ban đầu
    V = np.zeros(n_states)

    def q_values(s):
        q = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, s2, r, done in P[s][a]:
                q[a] += prob * (r + gamma * V[s2] * (not done))
        return q

    stable = False
    outer = 0
    while not stable:
        outer += 1

        # V(s) <- sum_{s'} P(s'|s,pi(s)) * [ R + gamma * V(s') ]
        # (hành động đã được chính sách CỐ ĐỊNH, không có phép max ở đây)
        while True:
            delta = 0.0
            for s in range(n_states):
                a = policy[s]
                v_new = 0.0
                for prob, s2, r, done in P[s][a]:
                    v_new += prob * (r + gamma * V[s2] * (not done))
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < theta:
                break

        # pi(s) <- argmax_a Q(s,a); nếu hành động thay đổi thì chính sách chưa ổn định
        stable = True
        for s in range(n_states):
            q = q_values(s)
            old_a = policy[s]
            best_a = q.argmax()
            # Chỉ đổi hành động khi nó tốt hơn HẲN (dung sai này tránh việc đổi qua
            # đổi lại mãi giữa các hành động có Q bằng nhau, chỉ lệch do sai số số thực)
            if q[best_a] > q[old_a] + 1e-10:
                policy[s] = best_a
                stable = False

    # - Mỗi bước improvement cho V_mới >= V_cũ ở mọi state.
    # - Chính sách chỉ đổi khi có hành động tốt hơn HẲN, nên mỗi lần đổi là
    #   chính sách được cải thiện thật sự.
    # - Số chính sách tất định là hữu hạn (6^500), và một chính sách không thể
    #   lặp lại vì giá trị chỉ tăng. Do đó vòng lặp phải kết thúc, và nó kết
    #   thúc tại chính sách tham lam theo chính giá trị của nó: tức là tối ưu.
    print("Policy iteration kết thúc sau", outer, "vòng lặp ngoài")
    return V, policy

env = gym.make("Taxi-v4")
V_vi, policy_vi = value_iteration(env)
V_pi, policy_pi = policy_iteration(env)

print("Mean Value VI", np.mean(V_vi))
print("Mean Value PI", np.mean(V_pi))
print("Different actions:", np.sum(policy_vi != policy_pi))