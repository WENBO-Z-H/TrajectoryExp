"""
读取Geolife数据集
"""
import math
import os
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 纬度范围
minLat = 39.7817
maxLat = 40.1075
# 经度范围
minLng = 116.1715
maxLng = 116.5728
# 网格维度
gridSize = 256


def ParseTime(time_str):
    time_list = time_str.split(':')
    seconds = int(time_list[0]) * 3600 + int(time_list[1]) * 60 + int(time_list[2])
    return seconds


def GetData(user_id):
    """
    读取Geolife 1.3数据集中特定id用户的前100条处于北京地区的轨迹数据
    :param user_id:
    :return: 三维列表表示的轨迹数据，第1维表示轨迹数目，第2维表示轨迹中的位置点数目，第3维表示纬度和经度
    """
    lat_str = []  # 纬度
    lng_str = []  # 经度
    time_str = []  # 时间

    path = "..\\Geolife Trajectories 1.3 - 副本" + "\\Data" + "\\" + user_id + "\\Trajectory"  # 000的路径
    plt_files = os.scandir(path)

    data = []  # 最终读取结果
    i = 0
    for item in plt_files:  # 每一个文件的绝对路径（前100条）
        if i >= 114:  # 114是因为其中有北京地区外的数据
            break
        else:
            i += 1
        path_item = path + "\\" + item.name
        with open(path_item, 'r+') as fp:
            for item in fp.readlines()[6:]:
                item_list = item.split(',')
                lat_str.append(item_list[0])
                lng_str.append(item_list[1])
                time_str.append(item_list[6])

        lat_float = [float(x) for x in lat_str]
        lng_float = [float(x) for x in lng_str]
        time_int = [ParseTime(x) for x in time_str]
        data.append(list(zip(lat_float, lng_float, time_int)))
        lat_str, lng_str, lat_float, lng_float = [], [], [], []

    # 每个位置点是否在北京地区内
    error = 0
    error_id = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if minLat <= data[i][j][0] <= maxLat and minLng <= data[i][j][1] <= maxLng:
                pass
            else:
                if i not in error_id:
                    error_id.append(i)
                error += 1
    print("不在北京区域内的点有", error, "个")
    print("分别属于路径: ", error_id)
    print("不删除北京外路径时，路径数量", len(data))
    error_id.reverse()
    for i in error_id:
        data.pop(i)
    print("删除北京外路径时，路径数量", len(data))

    return data


def GetCellID(lat, lng):
    """
    根据经纬度获得网格编码id
    :param lat: 位置点纬度
    :param lng: 位置点经度
    :return: 返回 0 <= x, y < 256
    """
    step = (maxLat - minLat) / gridSize
    x = int((lat - minLat) // step)
    step = (maxLng - minLng) / gridSize
    y = int((lng - minLng) // step)
    # 边界情况
    if x == 256:
        x = 255
    if y == 256:
        y = 255
    return x, y


def CalDist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def EncodeTraj(data):
    """
    轨迹iT2I编码
    :param data: 三维列表表示的轨迹数据，第1维表示轨迹数目，第2维表示轨迹中的位置点数目，第3维表示纬度和经度
    :return:
    """
    m_i1 = [[0 for i in range(gridSize)] for i in range(gridSize)]
    m_i2 = [[0 for i in range(gridSize)] for i in range(gridSize)]
    m_i3 = [[0 for i in range(gridSize)] for i in range(gridSize)]
    m_i4 = [[0 for i in range(gridSize)] for i in range(gridSize)]
    m_i5 = [[0 for i in range(gridSize)] for i in range(gridSize)]
    num_cell_points = [[0 for i in range(gridSize)] for i in range(gridSize)]

    # 第一次循环构造m_i1和m_i2
    print("第一次循环构造m_i1和m_i2")
    for traj in data:  # item是二维列表，表示多个位置点
        for point in traj:
            x, y = GetCellID(point[0], point[1])
            # print("x,y: ", x, y)
            num_cell_points[x][y] += 1
            m_i1[x][y] = 1  # 构建
            m_i2[x][y] += point[2]
            for i in range(gridSize):
                for j in range(gridSize):
                    if num_cell_points[i][j] == 0:  # 网格无点则保持为0
                        pass
                    else:  # 网格有点则取时间平均值
                        m_i2[i][j] = m_i2[i][j] / num_cell_points[i][j]

    # 第二次循环构造m_i3,m_i4,m_i5]
    print("第二次循环构造m_i3, m_i4, m_i5")
    cell_speed_list = [[[] for i in range(gridSize)] for i in range(gridSize)]  # 每个网格里的速度值，最后需要求中位数
    cell_acceleration_list = [[[] for i in range(gridSize)] for i in range(gridSize)]  # 每个网格里的加速度值，最后需要求中位数
    first_point = True  # 由于第一个点的速度和加速度都直接初始化为0，所以处理方式和其他点有区别
    last_speed = 0  # 上一个点的速度
    for traj_id in range(len(data)):  # item是二维列表，表示多个位置点
        print(traj_id)
        for point_id in range(len(data[traj_id])):
            x, y = GetCellID(data[traj_id][point_id][0], data[traj_id][point_id][1])
            # print("x,y: ", x, y)
            if first_point:
                m_i3[x][y] = 0
                m_i5[x][y] = 0
                last_speed = 0
            else:
                delta_t = data[traj_id][point_id][2] - data[traj_id][point_id - 1][2]  # 该点与上一点的时间差
                speed = CalDist(data[traj_id][point_id], data[traj_id][point_id - 1]) / delta_t  # 该点速度
                acceleration = (speed-last_speed) / delta_t  # 该点加速度
                cell_speed_list[x][y].append(speed)  # 添加到网格速度列表
                cell_acceleration_list[x][y].append(acceleration)    # 添加到网格加速度列表
                last_speed = speed  # 更新上一个点的速度
    M = [m_i1, m_i2, m_i3, m_i4, m_i5]
    M = torch.tensor(M)


if __name__ == '__main__':
    # user_id = "000"
    # data = GetData(user_id)
    # # print(data)
    # EncodeTraj(data)
    m_i1 = [[0 for i in range(gridSize)] for i in range(gridSize)]
    m_i2 = [[0 for i in range(gridSize)] for i in range(gridSize)]
    m_i3 = [[0 for i in range(gridSize)] for i in range(gridSize)]
    m_i4 = [[0 for i in range(gridSize)] for i in range(gridSize)]
    m_i5 = [[0 for i in range(gridSize)] for i in range(gridSize)]
    M = [m_i1, m_i2, m_i3, m_i4, m_i5]
    M = torch.tensor(M)
    print(M)
    print(M.shape)

