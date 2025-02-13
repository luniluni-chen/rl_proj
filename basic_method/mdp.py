import numpy as np

## 马尔可夫奖励过程
np.random.seed(0)
# 状态转移概率矩阵
P = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])



rewards = [-1, -2, -2, 10, 1, 0]  # 奖励函数


# 给定一条序列，计算某个状态的回报（return）
def compute_return(start_idx, chain, gamma=0.5):
    G = 0
    for idx in reversed(range(start_idx, len(chain))):
        G = gamma * G + rewards[chain[idx]-1]
    return G


chain = [1, 2, 3, 6]  # 状态序列 s1-s2-s3-s6
print(compute_return(0, chain))


