def compute_returns(rewards, gamma=1.0):
    returns = [0.0] * len(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):   
        G = rewards[t] + gamma * G             
        returns[t] = G
    return returns


# Ví dụ dùng thử
rewards = [0.0, 0.0, 1.0]
print(compute_returns(rewards))               
print(compute_returns(rewards, gamma=0.9))    