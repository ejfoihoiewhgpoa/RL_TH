def find_first_visits(episode):
    """Trả về dict {state: vị trí xuất hiện đầu tiên trong episode}."""
    first_visit = {}
    for t, (state, action, reward) in enumerate(episode):
        if state not in first_visit:          # chỉ ghi lần gặp đầu tiên
            first_visit[state] = t
    return first_visit


# Test 1: episode giả có state lặp lại để kiểm tra logic
fake_episode = [
    ((12, 10, False), 1, 0.0),   # t=0
    ((15, 10, False), 1, 0.0),   # t=1
    ((12, 10, False), 1, 0.0),   # t=2 (lặp lại state ở t=0)
    ((20, 10, False), 0, 1.0),   # t=3
    ((15, 10, False), 0, 0.0),   # t=4 (lặp lại state ở t=1)
]

first_visit = find_first_visits(fake_episode)
for state, t in first_visit.items():
    print(f"{state} xuất hiện đầu tiên ở t={t}")

# Test 2: episode thật (cần có env, generate_episode, stick_on_20_policy từ các bài trước)
# episode = generate_episode(env, stick_on_20_policy, seed=0)
# print(find_first_visits(episode))