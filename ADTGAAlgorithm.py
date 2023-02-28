import math
import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# matplotlib画图中中文显示会有问题，需要这两行设置默认字体


def Rotate(point, rotateAngle, rotatePoint):
    """
    Rotate a point at a specific angle along the center of rotation
    :param point: point that needs to be rotated
    :param rotateAngle: angle of the rotation
    :param rotatePoint: center of the rotation
    :return: coordinates of the rotated point
    """
    x, y = point
    x0, y0 = rotatePoint
    theta = rotateAngle
    x_ = (x - x0) * math.cos(theta) - (y - y0) * math.sin(theta) + x0
    y_ = (x - x0) * math.sin(theta) + (y - y0) * math.cos(theta) + y0
    return x_, y_


def GenerateDummy(realTrajectory, rotateAngle, rotatePoint):
    """
    Generate a fake trajectory by real trajectory、rotate angle and rotate point
    :param realTrajectory: real user trajectory
    :param rotateAngle: rotation angle ∈ (0,2π)
    :param rotatePoint: rotation point
    :return: dummy trajectory
    """
    points = [item[0:2] for item in realTrajectory]
    dummyTrajectory = [Rotate(point, rotateAngle, rotatePoint) for point in points]
    return dummyTrajectory


def DistanceBetweenTwoPoints(point1, point2):
    """
    Calculate the distance between two points (Euclidean Distance)
    :param point1: ...
    :param point2: ...
    :return: distance between two points
    """
    result = math.sqrt(pow((point1[0] - point2[0]), 2) + pow((point1[1] - point2[1]), 2))
    return result


def CalculateTDDForAll(trajectories, locationsInEachTra):
    """
    Calculate the Trajectories Distance Deviation (TDD) of current trajectories (including real and fake trajectory)
    :param trajectories: list of all current trajectories
    :param locationsInEachTra: locations in each trajectory
    :return: TDD value
    """
    n = len(trajectories) - 1  # number of dummy trajectories
    sum = 0
    for i in range(0, n + 1):  # 所有轨迹（包括真实轨迹）
        for k in range(i + 1, n + 1):  # 所有与i轨迹计算距离的轨迹
            for j in range(0, locationsInEachTra):  # 所有时刻
                sum = sum + DistanceBetweenTwoPoints(trajectories[i][j][:2], trajectories[k][j][:2])
    tdd = sum * (1 / locationsInEachTra) * (2 / n * (n + 1))
    return tdd


def CalTDDForNewDummy(oldTDD, trajectories, newTrajectory):
    """
    Calculate the TDD after adding a fake trajectory by averaging the previous TDD
    :param oldTDD: previous TDD without new fake trajectory
    :param trajectories: trajectories with out new fake trajectory
    :param newTrajectory: new generated trajectory
    :return: new TDD value
    """
    n = len(trajectories[0])
    locationsInEachTra = len(newTrajectory)
    sum = 0
    for item in trajectories:
        for j in range(0, locationsInEachTra):
            sum = sum + DistanceBetweenTwoPoints(item[j][:2], newTrajectory[j][:2])

    newTDD = n / (n + 1) * oldTDD + 1 / (n + 1) * sum
    return newTDD


def SatisfyTDDMetric(realTrajectory, dummys, newDummy, TDDMetric):
    """
    Judge if current trajectories satisfy TDD Metric
    :param realTrajectory: real trajectory
    :param dummys: fake trajectories
    :param newDummy: newly generated fake trajectory
    :param TDDMetric: TDD metric
    :return: whe ther the metric is satisfied or not
    """
    trajectories = realTrajectory + dummys
    # 1.是否满足新路径加入后TDD增加
    locationsInEachTra = len(realTrajectory[0])
    oldTDD = CalculateTDDForAll(trajectories, locationsInEachTra)
    newTDD = CalTDDForNewDummy(oldTDD, trajectories, newDummy)
    if newTDD <= oldTDD:
        return False
    # 2.TDD数值是否满足指标TDDMetric
    if newTDD > TDDMetric:
        return True
    else:
        return False


def Perturbation(dummy, k):
    pass


def ADTGAAlgorithm(realTrajectory, KMetric, TSDMetric, TDDMetric, LDMetric):
    """
    :param realTrajectory: real user trajectory
    :param KMetric: perturbation degree
    :param TSDMetric: Δt-Short term Disclosure
    :param TDDMetric: Trajectories Distance Deviation
    :param LDMetric: Long term Disclosure
    """
    dummys = []
    while True:
        candidate = []
        angles = list(range(1, 36))
        angles = [item * 1 / 18 * math.pi for item in angles]  # [10°, 20°, ... , 350°] （Radian system）

        for angle in angles:
            for rotatePoint in realTrajectory:
                dummy = GenerateDummy(realTrajectory, angle, rotatePoint)
                if not SatisfyTDDMetric(realTrajectory, dummys, newDummy, TDDMetric):  # 不满足指标
                    continue
                AdaptiveDummy()
                Perturbation(dummy, K)
                candidate.append(dummy)
        for dummy in candidate:
            ssd = 0
            ld = 0
        dummys.append()
        pass
    if 0:
        return dummys


def draw(trajectories):
    figure = plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=15, xmin=-15)
    plt.ylim(ymax=15, ymin=-15)
    axes = figure.add_subplot(1, 1, 1)
    color1 = '#FF0000'
    color2 = '#00FF00'
    area = np.pi * 4 ** 2  # 点面积
    for trajectory in trajectories:
        # 画折线图
        x = [i[0] for i in trajectory]
        y = [i[1] for i in trajectory]
        axes.plot(x, y)
        # 画起点、终点
        axes.scatter([x[0]], [y[0]], s=area, c=color1, alpha=1, label="起点")
        axes.scatter([x[-1]], [y[-1]], s=area, c=color2, alpha=1, label="终点")
    axes.set_aspect('equal')
    #plt.legend()
    plt.show()


if __name__ == '__main__':
    trajectory = [[0, 0, 0], [-2, 3, 1], [-2, 1, 2], [-2, 0, 3], [-5, 2, 4], [-5, -1, 5]]
    dummys = GenerateDummy(trajectory, 1/4*math.pi, (0, 0))
    draw([trajectory, dummys])
    print(dummys)
    # ADTGAAlgorithm(0, 0, 0, 0)
