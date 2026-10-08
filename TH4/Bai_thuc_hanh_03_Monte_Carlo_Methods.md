# BÀI THỰC HÀNH SỐ 3

## MONTE CARLO METHODS TRONG HỌC TĂNG CƯỜNG

**Học phần:** Học tăng cường – Reinforcement Learning  
**Ngôn ngữ lập trình:** Python  
**Nền tảng khuyến nghị:** Google Colab / Jupyter Notebook  
**Số lượng bài tập:** 36 bài  
**Hình thức:** Thực hành lập trình cá nhân  
**Hình thức nộp bài:** GitHub cá nhân  
**Folder nộp bài:** `Lab03/`

---

# 1. Ý tưởng của Lab03

Trong Lab02, sinh viên đã làm quen với environment, trajectory, reward, return, discount factor, policy, transition model, V(s), Q(s,a), Bellman backup và Value Iteration.

Lab03 chuyển sang **Monte Carlo Methods**, tức là học giá trị và policy từ **các episode đã quan sát được**, không cần biết trước đầy đủ transition probability hay reward model của môi trường.

Theo tài liệu môn học, Monte Carlo prediction ước lượng value function bằng **trung bình return** thu được từ các episode; có hai biến thể cơ bản là **First-Visit Monte Carlo** và **Every-Visit Monte Carlo**. Monte Carlo là phương pháp model-free và phù hợp với episodic tasks.

Mục tiêu của Lab03 là để sinh viên:

1. sinh episode từ một policy;
2. lưu đầy đủ trajectory;
3. tính return cho từng timestep;
4. cài đặt First-Visit MC prediction;
5. cài đặt Every-Visit MC prediction;
6. ước lượng V(s);
7. ước lượng Q(s,a);
8. xây dựng epsilon-greedy policy;
9. cài đặt Monte Carlo control;
10. đánh giá policy học được;
11. so sánh Monte Carlo với random policy;
12. quan sát sự hội tụ khi tăng số episode.

> **Tinh thần của bài:** Không dùng model transition. Sinh viên phải học từ episode thực tế do agent tương tác với environment.

---

# 2. Liên hệ với tài liệu

Trong tài liệu tham khảo, Monte Carlo được giới thiệu như một phương pháp dùng khi không biết trước transition và reward probabilities. Phương pháp chỉ cần các chuỗi state, action và reward, và được áp dụng cho episodic tasks.

Tài liệu phân biệt:

```text
First-Visit Monte Carlo
Every-Visit Monte Carlo
```

Trong đó:

- First-Visit MC chỉ sử dụng return tại lần xuất hiện đầu tiên của state trong một episode;
- Every-Visit MC sử dụng return ở mọi lần state xuất hiện trong episode.

Tài liệu cũng giới thiệu:

```text
Monte Carlo prediction
Monte Carlo control
Exploring Starts
On-policy MC control
Off-policy MC control
```

Lab03 tập trung trọng tâm vào:

```text
Monte Carlo prediction
First-Visit MC
Every-Visit MC
State-value estimation
Action-value estimation
epsilon-greedy policy
On-policy Monte Carlo control
```

Phần off-policy được để ở mức mở rộng.

---

# 3. Quy định chọn môi trường

Mỗi sinh viên chọn từ **1 đến 3 môi trường**.

## Môi trường chính

Một trong các môi trường episodic:

```text
Blackjack-v1
FrozenLake-v1
Taxi-v3
CliffWalking-v1
```

Trong đó `Blackjack-v1` là môi trường khuyến nghị vì phù hợp trực tiếp với Monte Carlo prediction/control.

## Môi trường phụ

Có thể chọn thêm:

```text
FrozenLake-v1
Taxi-v3
CliffWalking-v1
CartPole-v1
MountainCar-v0
```

Lưu ý:

- Monte Carlo cần episode kết thúc để tính return;
- nếu episode quá dài phải đặt `max_steps`;
- môi trường continuous state như CartPole/MountainCar khó lưu V(s) dạng bảng trực tiếp, nên chỉ dùng để minh họa trajectory/return hoặc phải tự rời rạc hóa state.

---

# 4. Khai báo môi trường đã chọn

Trong notebook:

```python
STUDENT_ID = "MSSV"
STUDENT_NAME = "Ho Ten"

MAIN_ENV = "Blackjack-v1"

OPTIONAL_ENVS = [
    "FrozenLake-v1",
]
```

Trong `README.md` ghi:

```text
Môi trường chính:
Môi trường phụ:
Lý do lựa chọn:
```

---

# 5. Cài đặt thư viện

Trên Google Colab:

```python
!pip install -q "gymnasium[toy-text,classic-control]"
```

Import:

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
```

Kiểm tra:

```python
print("Gymnasium:", gym.__version__)
print("NumPy:", np.__version__)
```

---

# 6. Cấu trúc nộp bài

```text
RL_MSSV_HoTen/
│
├── Lab01/
├── Lab02/
├── Lab03/
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   │   ├── bai01.py
│   │   ├── bai02.py
│   │   ├── ...
│   │   ├── bai36.py
│   │   ├── mc_utils.py
│   │   └── main.py
│   ├── notebooks/
│   │   └── Lab03_MSSV_HoTen.ipynb
│   ├── figures/
│   │   ├── mc_convergence.png
│   │   ├── first_vs_every_visit.png
│   │   ├── policy_performance.png
│   │   └── epsilon_comparison.png
│   ├── videos/
│   │   └── ...
│   └── data/
│       └── README.md
└── README.md
```

---

# 7. Các khái niệm cần dùng

## 7.1. Episode

Một episode được biểu diễn:

```text
S0, A0, R1, S1, A1, R2, ..., ST
```

Trong code có thể lưu:

```python
episode = [
    (state, action, reward),
    ...
]
```

## 7.2. Return

Tại timestep `t`:

```text
G_t = R_(t+1) + gamma R_(t+2) + gamma^2 R_(t+3) + ...
```

## 7.3. Monte Carlo Prediction

Ước lượng `V_pi(s)` bằng trung bình các return quan sát được sau khi state `s` xuất hiện trong episode.

## 7.4. First-Visit MC

Chỉ cập nhật state ở **lần xuất hiện đầu tiên trong episode**.

## 7.5. Every-Visit MC

Cập nhật ở **mọi lần state xuất hiện trong episode**.

## 7.6. Action-value function

Monte Carlo control sử dụng `Q(s,a)` thay vì chỉ V(s).

## 7.7. Epsilon-greedy policy

Với xác suất `1 - epsilon` chọn action tốt nhất; với xác suất `epsilon` khám phá action khác.

---

# 8. BÀI TẬP

## PHẦN A — Làm quen với episodic environment

### Bài 1. Tạo Blackjack-v1

Tạo:

```python
env = gym.make("Blackjack-v1")
```

Reset và in:

```text
observation
info
observation_space
action_space
```

### Bài 2. Phân tích observation

Với Blackjack, observation có dạng tuple. Sinh viên phải giải thích bằng comment:

```text
player sum
dealer showing card
usable ace
```

In ít nhất 10 observation từ nhiều episode.

### Bài 3. Phân tích action

In `env.action_space`, xác định ý nghĩa của từng action và tạo:

```python
ACTION_NAMES = {
    ...
}
```

### Bài 4. Một episode ngẫu nhiên

Chạy một episode bằng random policy. In từng bước:

```text
state
action
reward
next_state
terminated
truncated
```

### Bài 5. Lưu episode

Viết:

```python
def generate_episode(env, policy, seed=None):
    ...
```

Trả về `episode`, mỗi phần tử là:

```python
(state, action, reward)
```

---

## PHẦN B — Return từ episode

### Bài 6. Chuỗi reward

Từ episode vừa tạo, lấy `rewards = [...]`. In episode length, rewards và total reward.

### Bài 7. Tính return từ cuối episode

Viết:

```python
def compute_returns(rewards, gamma=1.0):
    ...
```

Trả về `G_0, G_1, ..., G_T`.

### Bài 8. Kiểm tra return

Tạo episode giả:

```python
rewards = [0, 0, 1]
```

Tính return với:

```text
gamma = 1.0
gamma = 0.9
gamma = 0.5
```

### Bài 9. Gắn return vào trajectory

Chuyển `(state, action, reward)` thành `(state, action, reward, G_t)` và in toàn bộ episode.

### Bài 10. So sánh gamma

Chạy cùng policy trong nhiều episode, tính mean `G_0` với:

```text
gamma = 0.5
gamma = 0.8
gamma = 0.9
gamma = 0.99
gamma = 1.0
```

Vẽ biểu đồ.

---

## PHẦN C — Monte Carlo Prediction cơ bản

### Bài 11. Policy cố định

Với Blackjack, tạo policy đơn giản:

```python
def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:
        return 0
    return 1
```

Sinh viên phải kiểm tra lại ý nghĩa action trước khi sử dụng.

### Bài 12. Sinh 100 episode

Dùng policy ở Bài 11, sinh 100 episode. Đếm `win`, `loss`, `draw` và tính tỉ lệ.

### Bài 13. Thu thập return theo state

Tạo:

```python
returns = defaultdict(list)
```

Với mỗi state trong episode, lưu return tương ứng. Chưa yêu cầu phân biệt first/every visit.

### Bài 14. Ước lượng V(s)

Tính:

```python
V[state] = np.mean(returns[state])
```

In value của ít nhất 10 state đã xuất hiện.

### Bài 15. Theo dõi một state cụ thể

Chọn một state, ví dụ:

```python
target_state = (20, 10, False)
```

Theo dõi estimate của V(state) sau:

```text
100
500
1000
5000
10000 episode
```

Vẽ convergence curve.

---

## PHẦN D — First-Visit Monte Carlo

### Bài 16. Phát hiện first visit

Với một episode, viết code xác định vị trí xuất hiện đầu tiên của mỗi state. Không sử dụng thư viện RL có sẵn.

### Bài 17. First-Visit MC Prediction

Viết:

```python
def first_visit_mc_prediction(
    env,
    policy,
    n_episodes,
    gamma=1.0
):
    ...
```

Trả về:

```python
V
returns_count
```

### Bài 18. Chạy First-Visit MC

Chạy với:

```text
100
1000
10000
50000 episode
```

In số state đã được ước lượng.

### Bài 19. Visualize value

Với Blackjack, chọn các state có `usable_ace = False` và tạo bảng/heatmap đơn giản theo `player sum` và `dealer showing card`.

---

## PHẦN E — Every-Visit Monte Carlo

### Bài 20. Every-Visit MC Prediction

Viết:

```python
def every_visit_mc_prediction(
    env,
    policy,
    n_episodes,
    gamma=1.0
):
    ...
```

### Bài 21. So sánh First-Visit và Every-Visit

Chạy cả hai với cùng policy, gamma, n_episodes và seed. So sánh V(s) tại ít nhất 10 state.

### Bài 22. Sai khác giữa hai phương pháp

Tính:

```python
abs(V_first[state] - V_every[state])
```

Tính mean absolute difference trên các state chung.

### Bài 23. Biểu đồ hội tụ

Chọn 3 state. Vẽ estimate V(s) của First-Visit MC và Every-Visit MC theo số episode.

Lưu:

```text
Lab03/figures/first_vs_every_visit.png
```

---

## PHẦN F — Action-value function Q(s,a)

### Bài 24. Lưu state-action-return

Thay vì `state -> return`, lưu `(state, action) -> return` bằng `defaultdict(list)`.

### Bài 25. Ước lượng Q(s,a)

Viết:

```python
def mc_action_value_prediction(
    env,
    policy,
    n_episodes,
    gamma=1.0
):
    ...
```

Trả về `Q`, trong đó `Q[state][action]`.

### Bài 26. Greedy action từ Q

Viết:

```python
def greedy_action(Q, state, n_actions):
    ...
```

Sử dụng `np.argmax(...)`.

---

## PHẦN G — Epsilon-greedy và Exploration

### Bài 27. Epsilon-greedy policy

Viết:

```python
def epsilon_greedy_action(
    Q,
    state,
    n_actions,
    epsilon,
    rng
):
    ...
```

### Bài 28. Kiểm tra epsilon-greedy

Với Q giả:

```python
Q_values = [1.0, 5.0]
```

Sinh 10000 action với `epsilon = 0.1`, đếm tần suất chọn action và nhận xét.

### Bài 29. So sánh epsilon

Thử:

```text
epsilon = 0.01
epsilon = 0.05
epsilon = 0.10
epsilon = 0.20
epsilon = 0.50
```

Vẽ tần suất chọn greedy action.

---

## PHẦN H — On-policy Monte Carlo Control

### Bài 30. Khởi tạo Q

Tạo:

```python
Q = defaultdict(
    lambda: np.zeros(env.action_space.n)
)
```

Tạo thêm `returns_sum`, `returns_count` hoặc dùng incremental mean.

### Bài 31. Incremental mean

Cài:

```text
N(s,a) = N(s,a) + 1
Q(s,a) = Q(s,a) + (G - Q(s,a)) / N(s,a)
```

Viết hàm cập nhật riêng.

### Bài 32. On-policy First-Visit MC Control

Viết:

```python
def on_policy_mc_control(
    env,
    n_episodes,
    gamma=1.0,
    epsilon=0.1,
    seed=42
):
    ...
```

Quy trình:

```text
Khởi tạo Q
    ↓
Sinh episode bằng epsilon-greedy
    ↓
Tính return
    ↓
First-visit update Q(s,a)
    ↓
Cập nhật policy epsilon-greedy
    ↓
Lặp lại
```

Trả về `Q`, `policy`, `episode_rewards`.

### Bài 33. Học policy cho Blackjack

Train ít nhất 100000 episode. Sau training, in policy tại các state ví dụ:

```text
(20, 10, False)
(18, 6, False)
(13, 2, False)
(18, 6, True)
```

Không hard-code action.

### Bài 34. Đánh giá policy đã học

Viết:

```python
def evaluate_policy(
    env,
    policy,
    n_episodes=10000,
    seed=123
):
    ...
```

Tính:

```text
win rate
loss rate
draw rate
mean reward
```

So sánh:

```text
random policy
fixed policy
MC learned policy
```

### Bài 35. Learning curve

Trong quá trình training, lưu `episode_rewards`, tính moving average với `window = 1000` và vẽ learning curve.

Lưu:

```text
Lab03/figures/mc_convergence.png
```

---

## PHẦN I — Mini-project

### Bài 36. Monte Carlo Agent hoàn chỉnh

Sinh viên xây dựng notebook/chương trình hoàn chỉnh gồm:

```text
1. Environment đã chọn
2. Observation/action space
3. Một episode mẫu
4. Trajectory
5. Reward
6. Return
7. Discounted return
8. Fixed policy
9. First-Visit MC Prediction
10. Every-Visit MC Prediction
11. So sánh hai prediction method
12. Q(s,a)
13. Epsilon-greedy
14. On-policy MC Control
15. Training curve
16. Learned policy
17. Evaluation
18. So sánh random/fixed/learned policy
19. Nhận xét
20. Kết luận
```

Nếu chọn thêm môi trường phụ, sinh viên phải áp dụng ít nhất `generate episode`, `compute return`, `MC prediction` trên môi trường đó.

---

# 9. Phần mở rộng tự chọn

## Mở rộng 1. Off-policy Monte Carlo

Tìm hiểu:

```text
target policy
behavior policy
importance sampling
```

Cài đặt off-policy MC prediction hoặc control.

## Mở rộng 2. Weighted Importance Sampling

So sánh ordinary importance sampling và weighted importance sampling.

## Mở rộng 3. Môi trường khác Blackjack

Thử MC control với:

```text
FrozenLake-v1
Taxi-v3
CliffWalking-v1
```

---

# 10. Cấu trúc `mc_utils.py`

Tạo:

```text
Lab03/src/mc_utils.py
```

Tối thiểu có:

```python
def generate_episode(...):
    ...

def compute_returns(...):
    ...

def first_visit_mc_prediction(...):
    ...

def every_visit_mc_prediction(...):
    ...

def mc_action_value_prediction(...):
    ...

def epsilon_greedy_action(...):
    ...

def on_policy_mc_control(...):
    ...

def evaluate_policy(...):
    ...
```

---

# 11. Skeleton chương trình

```python
import gymnasium as gym
import numpy as np

from collections import defaultdict


def generate_episode(env, policy, seed=None):
    # TODO
    pass


def compute_returns(rewards, gamma=1.0):
    # TODO
    pass


def first_visit_mc_prediction(
    env,
    policy,
    n_episodes,
    gamma=1.0
):
    # TODO
    pass


def every_visit_mc_prediction(
    env,
    policy,
    n_episodes,
    gamma=1.0
):
    # TODO
    pass


def epsilon_greedy_action(
    Q,
    state,
    n_actions,
    epsilon,
    rng
):
    # TODO
    pass


def on_policy_mc_control(
    env,
    n_episodes,
    gamma=1.0,
    epsilon=0.1,
    seed=42
):
    # TODO
    pass


def main():
    env = gym.make("Blackjack-v1")

    # TODO: prediction
    # TODO: control
    # TODO: evaluation

    env.close()


if __name__ == "__main__":
    main()
```

---

# 12. Yêu cầu trực quan hóa

Phải có tối thiểu:

```text
Lab03/figures/
├── mc_convergence.png
├── first_vs_every_visit.png
├── epsilon_comparison.png
└── policy_performance.png
```

Khuyến khích thêm:

```text
blackjack_value_no_usable_ace.png
blackjack_value_usable_ace.png
blackjack_policy.png
```

---

# 13. Nếu muốn render môi trường

Với environment hỗ trợ render:

```python
env = gym.make(
    "FrozenLake-v1",
    render_mode="rgb_array"
)
```

Dùng:

```python
frame = env.render()
```

và hiển thị bằng Matplotlib.

Với `Blackjack-v1`, trọng tâm của Lab03 là trajectory, state, action, reward và Monte Carlo estimate; không bắt buộc phải có video nếu environment không phù hợp cho việc render trực quan như game động.

---

# 14. Câu hỏi bắt buộc trong notebook

1. Monte Carlo method trong RL là gì?
2. Vì sao Monte Carlo được gọi là model-free?
3. Vì sao Monte Carlo cần episodic task?
4. Episode khác trajectory như thế nào trong bài này?
5. Return được tính khi nào?
6. Monte Carlo Prediction dùng để làm gì?
7. First-Visit MC khác Every-Visit MC thế nào?
8. Vì sao phải lấy trung bình nhiều return?
9. Khi số episode tăng, estimate V(s) thay đổi như thế nào?
10. V(s) khác Q(s,a) thế nào?
11. Vì sao MC control cần Q(s,a) thay vì chỉ V(s)?
12. Exploration là gì?
13. Exploitation là gì?
14. Epsilon-greedy cân bằng exploration/exploitation thế nào?
15. Khi epsilon quá lớn thì có vấn đề gì?
16. Khi epsilon quá nhỏ từ đầu thì có vấn đề gì?
17. Incremental mean có lợi gì so với lưu toàn bộ returns?
18. On-policy MC control nghĩa là gì?
19. Policy dùng để sinh episode có thay đổi trong quá trình học không?
20. Monte Carlo có bootstrap không?
21. MC phải đợi đến cuối episode vì sao?
22. Random policy và learned policy khác nhau ra sao về mean reward?
23. Learned policy có chắc chắn tối ưu không? Vì sao?
24. Số episode ảnh hưởng chất lượng estimate thế nào?
25. Monte Carlo khác Dynamic Programming ở điểm quan trọng nào?

---

# 15. README bắt buộc

```markdown
# Lab03 - Monte Carlo Methods

## Thông tin sinh viên

- Họ tên:
- MSSV:
- Lớp:

## Môi trường đã chọn

## Mục tiêu

## Cài đặt

## Cách chạy

## Episode và Return

## First-Visit MC

## Every-Visit MC

## Action-value Q(s,a)

## Epsilon-greedy

## On-policy MC Control

## Kết quả

## Learning Curve

## So sánh policy

## Nhận xét

## Tài liệu tham khảo
```

---

# 16. File requirements.txt

```text
gymnasium[toy-text,classic-control]
numpy
matplotlib
jupyter
```

---

# 17. Quy định code

Không được sử dụng thư viện gọi sẵn Monte Carlo prediction, Monte Carlo control hoặc epsilon-greedy agent.

Không hard-code V(s), Q(s,a), optimal policy.

Mỗi function phải có tên rõ nghĩa, tham số rõ nghĩa, giá trị trả về rõ nghĩa và docstring/comment.

---

# 18. Quy định GitHub

Ví dụ commit:

```text
Khoi tao Lab03
Them Blackjack environment
Generate Monte Carlo episodes
Compute discounted returns
Implement first visit MC
Implement every visit MC
Estimate action values
Add epsilon greedy policy
Implement on policy MC control
Add policy evaluation
Add learning curves
Complete Lab03 report
```

---

# 19. Thang điểm đề nghị

| Nội dung | Điểm |
|---|---:|
| Episode + trajectory | 0.75 |
| Return + gamma | 0.75 |
| Fixed policy + statistics | 0.75 |
| First-Visit MC Prediction | 1.25 |
| Every-Visit MC Prediction | 1.0 |
| So sánh First/Every Visit | 0.75 |
| Q(s,a) | 1.0 |
| Epsilon-greedy | 1.0 |
| On-policy MC Control | 1.5 |
| Evaluation + learning curve | 0.75 |
| GitHub + README + notebook | 0.5 |
| **Tổng** | **10.0** |

---

# 20. Điều kiện chấm

- Không sinh được episode: không tính phần Monte Carlo.
- Không tính return đúng từ cuối episode: yêu cầu sửa.
- Nhầm First-Visit và Every-Visit: không tính phần tương ứng.
- Hard-code V/Q/policy: không tính phần thuật toán.
- Dùng thư viện gọi sẵn MC control: không tính phần control.
- Không có exploration: không đạt điểm tối đa phần control.
- Chỉ chạy vài chục episode nhưng kết luận policy hội tụ: không đạt yêu cầu.
- Không có learning curve: trừ điểm.
- Không đánh giá policy trên episode mới: trừ điểm.
- Notebook không chạy được từ đầu đến cuối: phần kỹ thuật tối đa 50%.
- GitHub không truy cập được: có thể xem là chưa nộp.

---

# 21. Phân mức bài tập

## Mức 1 — Episode và Return

```text
Bài 1 → Bài 10
```

## Mức 2 — Monte Carlo Prediction

```text
Bài 11 → Bài 23
```

## Mức 3 — Q và Exploration

```text
Bài 24 → Bài 29
```

## Mức 4 — Monte Carlo Control

```text
Bài 30 → Bài 35
```

## Mức 5 — Tổng hợp

```text
Bài 36
```

---

# 22. Kết quả mong đợi

Sau Lab03, sinh viên phải hiểu được luồng:

```text
Policy
  ↓
Generate episode
  ↓
Trajectory
  ↓
Return G_t
  ↓
Average returns
  ↓
V(s) / Q(s,a)
  ↓
epsilon-greedy improvement
  ↓
Generate new episodes
  ↓
Improved policy
```

Sinh viên phải tự lập trình được tối thiểu:

```python
generate_episode()
compute_returns()
first_visit_mc_prediction()
every_visit_mc_prediction()
epsilon_greedy_action()
on_policy_mc_control()
evaluate_policy()
```

---

# 23. Liên hệ với Lab04

Monte Carlo phải đợi đến khi episode kết thúc mới có thể sử dụng return đầy đủ để update value.

Lab tiếp theo sẽ chuyển sang **Temporal-Difference Learning**, trong đó agent có thể cập nhật value ngay sau từng bước tương tác mà không cần chờ hết episode.

Luồng tiếp theo:

```text
Monte Carlo
    ↓
Temporal Difference
    ↓
SARSA
    ↓
Q-Learning
```

Đây là bước chuyển quan trọng từ:

```text
học từ full return
```

sang:

```text
học từ bootstrap estimate
```
