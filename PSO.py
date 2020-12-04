import math
import random


class PSO:
    def __init__(self, pop_size, dim, omega, c1, c2, x_max, x_min, y_max, y_min, z_max, z_min, fitnessFunction):
        self.c1 = c1  # 学习因子
        self.c2 = c2
        self.pop_size = pop_size  # 粒子群个体数
        self.dim = dim  # 变量数
        self.omega = omega  # 惯性因子
        self.xyz_max = [x_max, y_max, z_max]
        self.xyz_min = [x_min, y_min, z_min]
        self.v_max = [(x_max - x_min) * 0.05, (y_max - y_min) * 0.05,
                      (z_max - z_min) * 0.05]
        self.v_min = [-(x_max - x_min) * 0.05, -(y_max - y_min) * 0.05,
                      -(z_max - z_min) * 0.05]
        self.fitnessFunction = fitnessFunction
        print(self.fitnessFunction)
        self.position = [[]]  # 记录当前粒子位置
        self.position_history = [[]]  # 记录粒子历史移动信息，用于绘图
        self.best_position_index_history = []  # 记录粒子历史全局最优的索引信息，用于绘图
        self.speed = [[]]  # 记录当前粒子速度
        self.best_value = [[]]  # 记录全局最优
        self.value = [[]]  # 记录当前值

    def initial(self):
        for i in range(self.pop_size):
            x = []
            v = []
            for j in range(self.dim):
                x.append(math.floor(random.uniform(self.xyz_min[j], self.xyz_max[j]) * 100) / 100)
                v.append(math.floor(random.uniform(self.v_min[j], self.v_max[j]) * 100) / 100)
            self.position.append(x)
            self.speed.append(v)
            self.value.append(((self.fitness(x[0], x[1], x[2])), x[0], x[1], x[2]))
        self.value = self.value[1:]
        index = self.value.index(min(self.value))
        self.best_value.append(
            (self.value[index][0], self.position[index][0], self.position[index][1], self.position[index][2]))
        self.best_value = self.best_value[1:]
        self.position = self.position[1:]
        self.speed = self.speed[1:]

    def solving(self, times):
        for i in range(times):
            self.position_history.append([])
            pbest = self.value[self.value.index(min(self.value))]
            gbest = self.best_value[self.best_value.index(min(self.best_value))]
            for j in range(self.pop_size):
                x = []
                v = []
                for k in range(self.dim):
                    v.append(math.floor((self.omega * self.speed[j][k] + self.c1 * random.uniform(0, 1) * (
                            pbest[1 + k] - self.position[j][k]) + self.c2 * random.uniform(0, 1) * (
                                                 gbest[1 + k] - self.position[j][k])) * 100) / 100)
                    x.append(math.floor((self.position[j][k] + self.speed[j][k]) * 100) / 100)
                    if (v[k] < self.v_min[k]):
                        v[k] = self.v_min[k]
                    if (v[k] > self.v_max[k]):
                        v[k] = self.v_max[k]
                    if (x[k] < self.xyz_min[k]):
                        x[k] = self.xyz_min[k]
                    if (x[k] > self.xyz_max[k]):
                        x[k] = self.xyz_max[k]
                self.position[j] = x
                self.position_history[i].append(x)  # 记录历史位置信息
                self.speed[j] = v
                self.value[j] = (
                    self.fitness(self.position[j][0], self.position[j][1], self.position[j][2]), self.position[j][0],
                    self.position[j][1], self.position[j][2])
            index = self.value.index(min(self.value))
            self.best_value.append(
                (self.value[index][0], self.position[index][0], self.position[index][1], self.position[index][2]))
            self.best_position_index_history.append(index)
        print("yes")

    def fitness(self, x1, x2, x3):  # 适应度函数
        return eval(self.fitnessFunction)/100  # 识别函数
        
    def returnbest(self):
        return self.best_value

    def rerturn_position_history(self):
        return self.position_history

    def return_best_position_index_history(self):
        return self.best_position_index_history
