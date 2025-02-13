"""
不带有状态的简单环境-多臂老虎机
"""

import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """ 伯努利多臂老虎机，输入K表示拉杆数量，每根杆p概率获得奖励1，1-p概率下获得奖励0 """
    def __init__(self, K):
        self.K = K
        self.probs = np.random.uniform(size=self.K)  # 随机生成K个0-1的数，表示每个杆的获奖概率p
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的索引
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率

    def step(self, k):
        # 选择第k个拉杆后，根据概率p返回1（获奖）或0（未获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class Solver:
    """ 多臂老虎机算法的基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 统计每个杆的拉动次数
        self.regret = 0.0  # 当前步的懊悔值
        self.actions = []   # 动作列表
        self.regrets = []   # 懊悔值列表

    def update_regret(self, k):
        # 当前拉动第k个杆
        self.regret += self.bandit.best_prob - self.bandit.probs[k]  # 服从伯努利分布（期望等于p）
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    """ epsilon贪婪算法，继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super().__init__(bandit)
        self.epsilon = epsilon
        self.init_prob = init_prob
        self.estimates = np.array([init_prob]*self.bandit.K)  # 所有杆的期望奖励的估值

    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的杆

        r = self.bandit.step(k)  # 拉动第k杆后，老虎机给的奖励
        self.estimates[k] += 1./(self.counts[k]+1)*(r-self.estimates[k])  # 更新期望奖励估计值
        return k


class DecayingEpsilonGreedy(Solver):
    """ epsilon衰减的greedy策略算法 """
    def __init__(self, bandit, init_prob=1.0):
        super().__init__(bandit)
        self.estimates = np.array([init_prob]*self.bandit.K)
        self.total_cnt = 0

    def run_one_step(self):
        self.total_cnt += 1
        if np.random.rand() < 1/self.total_cnt:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1./(self.counts[k]+1)*(r-self.estimates[k])
        return k


def plot_results(solvers, solver_names):
    """ 绘制累计懊悔值-时间图，solvers为各种策略算法实例的列表 """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])

    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title("%d-armed bandit" % solvers[0].bandit.K)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ## 生成老虎机
    np.random.seed(1)
    K = 10
    bandit = BernoulliBandit(K)
    rounded_probs = [round(prob, 4) for prob in bandit.probs]
    print("随机生成%d臂伯努利老虎机" % K)
    print("%d个杆的获奖概率分别为" % K, rounded_probs)
    print("最大获奖概率的杆为%d号，其获奖概率为%.4f" % (bandit.best_idx, bandit.best_prob))

    ## 固定epsilon的greedy贪婪算法
    # np.random.seed(1)
    # epsilon_greedy_solver = EpsilonGreedy(bandit)
    # epsilon_greedy_solver.run(5000)
    # print(epsilon_greedy_solver.regrets[-1])
    # plot_results([epsilon_greedy_solver], ['epsilon_greedy'])

    ## 固定epsilon的greedy贪婪算法：epsilon超参
    # np.random.seed(0)
    # epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    # epsilon_greedy_solver_list = [
    #     EpsilonGreedy(bandit, eps) for eps in epsilons
    # ]
    # epsilon_greedy_solver_names = [
    #     "epsilon=%s" % eps for eps in epsilons
    # ]
    # for solver in  epsilon_greedy_solver_list:
    #     solver.run(5000)
    # plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

    ## 测试epsilon衰减的greedy算法
    np.random.seed(1)
    decaying_epsilon_greedy = DecayingEpsilonGreedy(bandit)
    decaying_epsilon_greedy.run(5000)
    print(decaying_epsilon_greedy.regret)
    plot_results([decaying_epsilon_greedy], ["decaying_epsilon"])







