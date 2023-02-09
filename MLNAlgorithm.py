import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# matplotlib画图中中文显示会有问题，需要这两行设置默认字体


class Dummy:
    def __init__(self, x=0, y=0, t=0):
        self.x = x
        self.y = y
        self.t = t


def Position(neighborhoodSize, dummyX, dummyY, othersLocations):
    sum = 0
    for eachList in othersLocations:
        for (x, y, t) in eachList:  # 遍历每一个真实位置
            if dummyX - neighborhoodSize < x < dummyX + neighborhoodSize \
                    and dummyY - neighborhoodSize < y < dummyY + neighborhoodSize:  # 如果其在虚拟位置邻域
                sum = sum + 1
    return sum


def MovingInNeighborhood(aveP, m, n):
    repeatCount = 3
    dummyPretemp = Dummy()  # 前一时刻用户位置及时间信息
    dummyNexttemp = Dummy()  # 后一时刻用户位置及时间信息
    dummys = [[dummyNexttemp.x, dummyNexttemp.y, dummyNexttemp.t]]

    i = 0
    while True:
        dummyPretemp = dummyNexttemp
        # 生成下一位置及时间信息
        dummyNexttemp.x = random.randint(dummyPretemp.x - m, dummyPretemp.x + m)
        dummyNexttemp.y = random.randint(dummyPretemp.y - m, dummyPretemp.y + m)
        dummyNexttemp.t = dummyPretemp.t + 1

        if Position(2, dummyNexttemp.x, dummyNexttemp.y, []) > aveP:
            if repeatCount > 0:
                repeatCount = repeatCount - 1
                continue
            else:
                repeatCount = 0
        dummys.append([dummyNexttemp.x, dummyNexttemp.y, dummyNexttemp.t])

        i = i + 1
        if i >= n:
            break
    return dummys


def draw(dummys):
    plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=15, xmin=-15)
    plt.ylim(ymax=15, ymin=-15)
    color1 = '#FF0000'
    color2 = '#00FF00'
    area = np.pi * 4 ** 2  # 点面积
    # 画折线图
    x = [i[0] for i in dummys]
    y = [i[1] for i in dummys]
    plt.plot(x, y)
    # 画起点、终点
    plt.scatter([x[0]], [y[0]], s=area, c=color1, alpha=1, label="起点")
    plt.scatter([x[-1]], [y[-1]], s=area, c=color2, alpha=1, label="终点")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dummys = MovingInNeighborhood(5, 3, 5)
    print(dummys)
    draw(dummys)
