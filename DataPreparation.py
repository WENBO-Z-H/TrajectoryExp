"""
读取Geolife数据集
"""
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# torch.set_printoptions(threshold=np.inf)
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
    """
    把"hh:mm:ss"格式的时间转换为秒数
    :param time_str: 字符串格式的时间
    :return: 转换成的秒数
    """
    time_list = time_str.split(':')
    seconds = int(time_list[0]) * 3600 + int(time_list[1]) * 60 + int(time_list[2])
    return seconds


def VectorUnitize(vector):
    """
    向量单位化
    :param vector: 含2个元素的向量
    :return: 单位化的向量
    """
    # print("vector: (", vector[0], vector[1], ")")
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2)

    if length == 0:
        return 0, 0
    else:
        return vector[0] / length, vector[1] / length


def CalAverage(data_list):
    """
    计算一个方向向量列表的平均值
    :param data_list: 目标列表
    :return: 平均方向（x,y组成的向量）
    """
    # print("CalAverage")
    length = len(data_list)
    if length == 0:  # 空列表的平均数按0计算
        return 0, 0
    x, y = 0, 0
    for item in data_list:
        x += item[0]
        y += item[1]
    x /= length
    y /= length
    return x, y


def CalMedian(data_list):
    """
    计算一个列表的中位数
    :param data_list: 目标列表
    :return: 目标列表的中位数
    """
    # print("CalMedian")
    length = len(data_list)
    if length == 0:  # 空列表中位数按0计算
        return 0
    data_list = sorted(data_list)
    if len(data_list) % 2 == 0:  # 偶数
        return 0.5 * (data_list[length // 2] + data_list[length // 2 - 1])
    else:
        return data_list[length // 2]


def VecToAngle(dir_vec):
    """
    计算一个方向向量与正北方向的夹角
    :param dir_vec: 含 x,y 2个元素的方向向量
    :return: 其与正北方向夹角的余弦值[-1, 1]
    """
    north_vec = (0, 1)
    dir_vec = VectorUnitize(dir_vec)  # 化为单位向量
    len_north_vec = 1
    len_dir_vec = 1
    cos_theta = (north_vec[0] * dir_vec[0] + north_vec[1] * dir_vec[1]) / (len_north_vec * len_dir_vec)
    return cos_theta


def GetData(user_id):
    """
    读取Geolife 1.3数据集中特定id用户的前100条处于北京地区的轨迹数据
    :param user_id:
    :return: 三维列表表示的轨迹数据，第1维表示轨迹数目，第2维表示轨迹中的位置点数目，第3维表示纬度、经度和时间
    """
    lat_str = []  # 纬度
    lng_str = []  # 经度
    time_str = []  # 时间

    path = "../Geolife-Trajectories-1.3" + "/Data" + "/" + user_id + "/Trajectory"  # 000的路径
    # path = "..\\Geolife Trajectories 1.3 - 副本" + "\\Data" + "\\" + user_id + "\\Trajectory"  # 000的路径
    plt_files = os.scandir(path)
    plt_files = os.listdir(path)
    plt_files.sort()

    data = []  # 最终读取结果
    i = 0
    for item in plt_files:  # 每一个文件的绝对路径（前100条）
        if i >= 114:  # 114是因为其中有北京地区外的数据
            break
        else:
            i += 1
        path_item = path + "/" + item
        # path_item = path + "/" + item.name
        # path_item = path + "\\" + item.name
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
    """
    计算两位置点间欧式距离
    :param point1:
    :param point2:
    :return: 距离
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def EncodeTrajWithIT2I(data):
    """
    轨迹iT2I编码
    :param data: 二维列表表示的轨迹数据，第1维表示轨迹中的位置点数目，第2维表示纬度和经度
    :return: 编码后的 5 * gridSize * gridSize 的矩阵组成的tensor
    """
    m_i1 = [[0 for i in range(gridSize)] for i in range(gridSize)]  # 是否有位置点
    m_i2 = [[0 for i in range(gridSize)] for i in range(gridSize)]  # 时间
    m_i3 = [[0 for i in range(gridSize)] for i in range(gridSize)]  # 速度
    m_i4 = [[0 for i in range(gridSize)] for i in range(gridSize)]  # 方向
    m_i5 = [[0 for i in range(gridSize)] for i in range(gridSize)]  # 加速度

    # 第一次遍历构造m_i1和m_i2 (一个路径遍历和一个网格遍历)
    num_cell_points = [[0 for i in range(gridSize)] for i in range(gridSize)]
    #print("第一次遍历构造m_i1和m_i2")
    for point in data:
        # print(point)
        x, y = GetCellID(point[0], point[1])
        # print("x,y: ", x, y)
        num_cell_points[x][y] += 1
        m_i1[x][y] = 1
        m_i2[x][y] += point[2]
    for i in range(gridSize):
        for j in range(gridSize):
            if num_cell_points[i][j] == 0:  # 网格无点则保持为0
                pass
            else:  # 网格有点则取时间平均值
                m_i2[i][j] = m_i2[i][j] / num_cell_points[i][j]

    # 第二次循环构造m_i3,m_i4,m_i5
    cell_speed_list = [[[] for i in range(gridSize)] for i in range(gridSize)]  # 每个网格里的速度值，最后需要求中位数
    cell_direction_list = [[[] for i in range(gridSize)] for i in range(gridSize)]  # 每个网格里的方向值，最后需要求平均数
    cell_acceleration_list = [[[] for i in range(gridSize)] for i in range(gridSize)]  # 每个网格里的加速度值，最后需要求中位数
    #print("第二次循环构造m_i3, m_i4, m_i5")
    first_point = True  # 由于第一个点的速度和加速度都直接初始化为0，所以处理方式和其他点有区别
    last_speed = 0  # 上一个点的速度
    for point_id in range(len(data)):
        x, y = GetCellID(data[point_id][0], data[point_id][1])
        # print("x,y: ", x, y)
        if first_point:
            m_i3[x][y] = 0
            m_i4[x][y] = 0
            m_i5[x][y] = 0
            last_speed = 0
            first_point = False
        else:
            delta_t = data[point_id][2] - data[point_id - 1][2]  # 该点与上一点的时间差
            speed = CalDist(data[point_id], data[point_id - 1]) / delta_t  # 该点速度
            direction = VectorUnitize([data[point_id][0] - data[point_id - 1][0],
                                       data[point_id][1] - data[point_id - 1][1]])  # 该点前进方向
            acceleration = (speed-last_speed) / delta_t  # 该点加速度
            cell_speed_list[x][y].append(speed)  # 添加到网格速度列表
            cell_direction_list[x][y].append(direction)  # 添加到网格方向列表
            cell_acceleration_list[x][y].append(acceleration)    # 添加到网格加速度列表
            last_speed = speed  # 更新上一个点的速度

    #print("更新3D矩阵")
    for i in range(gridSize):
        for j in range(gridSize):
            if num_cell_points[i][j] == 0:  # 网格无点则保持为0
                pass
            else:  # 网格有点则取速度和加速度的中值，方向的平均值
                pass
                m_i3[i][j] = CalMedian(cell_speed_list[i][j])
                m_i4[i][j] = VecToAngle(CalAverage(cell_direction_list[i][j]))
                m_i5[i][j] = CalMedian(cell_acceleration_list[i][j])

    M = [m_i1, m_i2, m_i3, m_i4, m_i5]
    M = torch.tensor(M, dtype=torch.float32)

    return M


def EncodeInd(data):
    """
    编码个体特征
    :param data: 二维列表表示的轨迹数据，第1维表示轨迹中的位置点数目，第2维表示纬度和经度
    :return: 编码后的含4个元素的向量，分别为轨迹距离、总用时、起始时间、平均速度
    """
    distance = 0
    for i in range(1, len(data)):
        distance += CalDist(data[i], data[i-1])
    total_time = data[-1][2] - data[0][2]
    start_time = data[0][2]
    avg_speed = distance / total_time
    return torch.tensor([distance, total_time, start_time, avg_speed], dtype=torch.float32)


if __name__ == '__main__':
    user_id = "000"
    data = GetData(user_id)
    exit(0)
    M = EncodeTrajWithIT2I(data[0])
    Ind = EncodeInd(data[0])
    print(M.shape)

