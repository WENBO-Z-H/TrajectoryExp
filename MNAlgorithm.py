"""
使用MN算法生成虚拟轨迹并绘制图像
"""

import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# matplotlib画图中中文显示会有问题，需要这两行设置默认字体

# 纬度
minLat = 39.7817
maxLat = 40.1075
# 经度
minLng = 116.1715
maxLng = 116.5728


class DummyPoint:
    def __init__(self, lng=0, lat=0, t=0):
        """
        虚拟轨迹点
        :param lng: 经度
        :param lat: 纬度
        :param t:时间戳
        """
        self.lng = lng
        self.lat = lat
        self.t = t

    def WithinBounds(self):
        """
        该点是否在规定区域内
        :return: 是或否
        """
        if minLat <= self.lat <= maxLat and minLng <= self.lng <= maxLng:
            return True
        return False


def MovingInNeighborhood(m, n):
    """
    :param m: 每次移动的幅度
    :param n: 距离长度
    :return: 虚拟轨迹
    """
    dummyPretemp = DummyPoint()  # 前一时刻用户位置及时间信息
    dummyNexttemp = DummyPoint()  # 后一时刻用户位置及时间信息
    dummyNexttemp.lng = random.uniform(minLng, maxLng)
    dummyNexttemp.lat = random.uniform(minLat, maxLat)
    dummys = [[dummyNexttemp.lng, dummyNexttemp.lat, 0]]  # 初始状态，轨迹列表只有初始点

    i = 0
    dummyPretemp = dummyNexttemp
    while i < n:
        # 生成下一位置及时间信息
        dummyNexttemp.lng = random.uniform(dummyPretemp.lng - m, dummyPretemp.lng + m)
        dummyNexttemp.lat = random.uniform(dummyPretemp.lat - m, dummyPretemp.lat + m)
        dummyNexttemp.t = dummyPretemp.t + 1
        if dummyNexttemp.WithinBounds():  # 在规定区域内才添加到列表，否则重新生成
            dummyPretemp = dummyNexttemp
            dummys.append([dummyNexttemp.lng, dummyNexttemp.lat, dummyNexttemp.t])
            i = i + 1

    return dummys


def draw(dummys):
    """
    绘制虚拟轨迹
    :param dummys: 虚拟轨迹(列表格式)
    """
    plt.figure()
    plt.xlabel('Lng')
    plt.ylabel('Lat')
    plt.xlim(xmax=116.5728, xmin=116.1715)
    plt.ylim(ymax=40.1075, ymin=39.7817)
    color1 = '#FF0000'
    color2 = '#00FF00'
    area = np.pi * 4**2  # 点面积
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
    dummys = MovingInNeighborhood(0.0001, 10)  # 生成一条虚拟轨迹，每次步长限制在0.0001*2，含10个位置点
    print(dummys)
    print(len(dummys))
    draw(dummys)
