"""
环境未知（model-free）的两种经典算法：Sarsa Q-learning，均基于时序差分（估计策略的价值函数）算法
蒙特卡洛的增量更新，需要一条序列结束获得Gt，时序差分只需要当前步结束
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalkingEnv:
    """ 交互版的悬崖散步环境 """
    def __init__(self, nrow, ncol):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0
        self.y = self.nrow-1

    def step(self, action):
        change = [[0,-1], [0,1], [-1,0], [1,0]]
        self.x = min(self.ncol-1, max(0, self.x+change[action][0]))
        self.y = min(self.nrow-1, max(0, self.y+change[action][1]))
        next_state = self.y*self.ncol+self.x

        reward = -1
        done = False
        if self.y == self.nrow-1 and self.x > 0:
            done = True
            if self.x != self.ncol-1:
                reward = -100
        return next_state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.nrow-1
        return self.y*self.ncol+self.x


class Sarsa:
    """ Sarsa算法，on-policy算法 """
    def __init__(self, nrow, ncol, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow*ncol, n_action])
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_action = n_action

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        Q_max = max(self.Q_table[state])
        a = [0]*self.n_action
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r+self.gamma*self.Q_table[s1, a1]-self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha*td_error

class Qlearning:
    """ Qlearning算法，off-policy """

    def __init__(self, nrow, ncol, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_action = n_action

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        Q_max = max(self.Q_table[state])
        a = [0] * self.n_action
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    # off-policy，不需要传入真实的a1，直接max策略代替
    def update(self, s0, a0, r, s1):
        td_error = r+self.gamma*max(self.Q_table[s1])-self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha*td_error


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

# Sarsa
# if __name__ == "__main__":
#     ncol = 12
#     nrow = 4
#     env = CliffWalkingEnv(nrow, ncol)
#     np.random.seed(0)
#     epsilon = 0.1
#     alpha = 0.1
#     gamma = 0.9
#     agent = Sarsa(nrow, ncol, epsilon, alpha, gamma)
#     num_episodes = 500  # 智能体在环境中运行的序列的数量
#
#     return_list = []    # 记录每条序列的回报
#     for i in range(10):  # 显示10个进度条
#         with tqdm(total=int(num_episodes/10), desc="Iteration: %d" % i) as pbar:
#             for i_episode in range(int(num_episodes/10)):   # 每个进度条的序列数
#                 episode_return = 0
#                 state = env.reset()
#                 action = agent.take_action(state)   # 初始化执行一次，epsilon贪婪算法获取动作a0
#                 done = False
#                 while not done:
#                     next_state, reward, done = env.step(action)  # 采取动作a0后，环境给出新状态和奖励
#                     next_action = agent.take_action(next_state)   # epsilon贪婪算法获取动作a1，表明是on-policy
#                     episode_return += reward    # 为了展示，不乘折扣因子
#                     agent.update(state, action, reward, next_state, next_action)
#                     state = next_state
#                     action = next_action      # a1赋给a0
#                 return_list.append(episode_return)
#                 if (i_episode+1)%10 == 0:
#                     pbar.set_postfix({
#                         'episode':
#                         '%d' % (num_episodes/10*i+i_episode+1),
#                         'return':
#                         '%.3f' % (np.mean(return_list[-10:]))
#                     })
#                 pbar.update(1)
#
#     episodes_list = list(range(len(return_list)))
#     plt.plot(episodes_list, return_list)
#     plt.xlabel('Episodes')
#     plt.ylabel('Returns')
#     plt.title('Sarsa on {}'.format('Cliff Walking'))
#     plt.legend()
#     plt.show()
#
#     action_meaning = ['^', 'v', '<', '>']
#     print('Sarsa算法最终收敛得到的策略为：')
#     print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

# Q-learning
if __name__ == "__main__":
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(nrow, ncol)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = Qlearning(nrow, ncol, epsilon, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []    # 记录每条序列的回报
    for i in range(10):  # 显示10个进度条
        with tqdm(total=int(num_episodes/10), desc="Iteration: %d" % i) as pbar:
            for i_episode in range(int(num_episodes/10)):   # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)  # epsilon贪婪算法获取动作a0,放到循环里面
                    next_state, reward, done = env.step(action)  # 采取动作后，环境给出新状态和奖励
                    episode_return += reward    # 为了展示，不乘折扣因子
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode+1)%10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes/10*i+i_episode+1),
                        'return':
                        '%.3f' % (np.mean(return_list[-10:]))
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Q-learning on {}'.format('Cliff Walking'))
    plt.legend()
    plt.show()

    action_meaning = ['^', 'v', '<', '>']
    print('Q-learning算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
