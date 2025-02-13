"""
动态规划算法求解最佳策略，适用于环境已知的情况（model-based，奖励函数和转移概率已知）
"""

import copy


class CliffWalkingEnv:
    """
    悬崖漫步环境，起点在左下角，目标在右下角，最后一行除了起点和目标都为悬崖
    由于环境已知，所以不需要和agent交互
    """
    def __init__(self, nrow=4, ncol=12):
        self.nrow = nrow
        self.ncol = ncol
        self.P = self.createP()

    def createP(self):
        P = [[[] for j in range(4)] for i in range(self.nrow*self.ncol)]
        change = [[0,-1], [0,1], [-1,0], [1,0]]  # 上下左右
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if i == self.nrow-1 and j > 0:  # 到了悬崖，或者目标点，无法交互，任何动作奖励为0
                        P[i*self.ncol+j][a] = [(1, i*self.ncol+j, 0, True)]
                        continue

                    next_x = min(self.ncol-1, max(0, j+change[a][0]))   # 碰到边界，状态不发生改变
                    next_y = min(self.nrow-1, max(0, i+change[a][1]))
                    next_state = next_y*self.ncol+next_x
                    reward = -1
                    done = False

                    if next_y == self.nrow-1 and next_x > 0:
                        done = True
                        if next_x != self.ncol-1:  # 下一个位置在悬崖，不在目标点
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


class PolicyIteration:
    """ 策略迭代，包含策略评估与策略提升（策略通常快速收敛，但每次评估计算量大） """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0]*self.env.nrow*self.env.ncol  # 初始化状态价值函数
        self.pi = [[0.25, 0.25, 0.25, 0.25] for _ in range(self.env.nrow*self.env.ncol)]  # 初始化随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.nrow*self.env.ncol
            for s in range(self.env.nrow*self.env.ncol):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:    # 对应qsa公式里对vs期望求和
                        p, next_state, r, done = res
                        qsa += p*r+self.gamma*p*self.v[next_state]*(1-done)  # 该类环境奖励跟下一状态有关，所以奖励也需要乘转移概率
                    qsa_list.append(self.pi[s][a]*qsa)
                new_v[s] = sum(qsa_list)    # vs等于qsa的期望求和
                max_diff = max(max_diff, abs(new_v[s]-self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("策略评估经过%d轮后完成" % cnt)

    def policy_improvement(self):
        for s in range(self.env.nrow*self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p*r+self.gamma*p*self.v[next_state]*(1-done)
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1/cntq if q == maxq else 0 for q in qsa_list]  # 最大的qsa的平分概率
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if new_pi==old_pi:
                break


class ValueIteration:
    """
    价值迭代，隐式策略（价值函数收敛较慢，但单次迭代更快）
    说明，策略提升可以在策略未完全评估的情况下进行
    """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0]*self.env.nrow*self.env.ncol
        self.theta = theta
        self.gamma = gamma
        self.pi = [None for _ in range(self.env.nrow*self.env.ncol)]

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0]*self.env.nrow*self.env.ncol
            for s in range(self.env.nrow*self.env.ncol):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p*r+self.gamma*p*self.v[next_state]*(1-done)
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s]-self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("价值迭代经过%d轮" % cnt)
        self.get_policy()

    def get_policy(self):
        for s in range(self.env.nrow*self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p*r+self.gamma*p*self.v[next_state]*(1-done)
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1/cntq if q == maxq else 0 for q in qsa_list]


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print("%6.6s" % ('%.3f' % agent.v[i*agent.env.ncol+j]), end=" ")
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if i*agent.env.ncol+j in disaster:
                print("****", end=" ")
            elif i*agent.env.ncol+j in end:
                print("EEEE", end=" ")
            else:
                a = agent.pi[i*agent.env.ncol+j]
                pi_str = ""
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=" ")
        print()


# if __name__ == "__main__":
#     env = CliffWalkingEnv()
#     action_meaning = ['^', 'v', '<', '>']
#     theta = 0.01
#     gamma = 0.9
#     # agent = PolicyIteration(env, theta, gamma)
#     # agent.policy_iteration()
#     agent = ValueIteration(env, theta, gamma)
#     agent.value_iteration()
#     print_agent(agent, action_meaning, list(range(37, 47)), [47])


import gym
env = gym.make("FrozenLake-v1")  # 创建冰湖环境
env = env.unwrapped  # 解封装才能访问状态转移矩阵P
env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)

for a in env.P[14]:  # 查看目标左边一格的状态转移信息
    print(env.P[14][a])

# 这个动作意义是Gym库针对冰湖环境事先规定好的
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])