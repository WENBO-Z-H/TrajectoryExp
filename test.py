import math


# 纬度范围
minLat = 39.7817
maxLat = 40.1075
# 经度范围
minLng = 116.1715
maxLng = 116.5728
# 网格维度
gridSize = 256

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



def VectorUnitization(vector):
    """
    向量单位化
    :param vector: 含2个元素的向量
    :return: 单位化的向量
    """
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    return (vector[0] / length, vector[1] / length)


# print(GetCellID(40.1075, 116.5728))
print(VectorUnitization([2, 2]))
