import os
import numpy as np
import time
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84

minLat = 39.7817
maxLat = 40.1075
minLng = 116.1715
maxLng = 116.5728

cell = 256


# 获得子文件
def get_sub_paths(path):
    all_ = []
    list_ = os.listdir(path)
    for sub_ in list_:
        new_ = path + "\\" + sub_
        all_.append(new_)
    return all_


# 获得 labels
def get_labels(path):
    label_data = []  # [start time, end time, label]
    f = open(path)
    data = f.readlines()
    f.close()
    for i in range(1, len(data)):
        line_ = data[i][0:len(data[i]) - 1].split(sep="\t")
        t1 = time.strptime(line_[0], "%Y/%m/%d %H:%M:%S")
        t2 = time.strptime(line_[1], "%Y/%m/%d %H:%M:%S")
        label_data.append([time.mktime(t1), time.mktime(t2), line_[2]])
    return label_data


# 轨迹 [lat, lng, time]
def get_traj(path):
    f = open(path)
    trs = f.readlines()[6:]
    f.close()
    traj_each = []
    for traj in trs:
        line_ = traj[:len(traj) - 1].split(sep=',')
        t = time.strptime(line_[5] + " " + line_[6], "%Y-%m-%d %H:%M:%S")
        lat = round(float(line_[0]), 6)
        lng = round(float(line_[1]), 6)
        stamp = time.mktime(t)
        traj_each.append([lat, lng, stamp])
    return traj_each


def get_traj_2(path):
    f = open(path)
    data = f.readlines()
    f.close()
    gps_data = []
    for line_ in data:
        line = line_[:len(line_) - 1].split(',')
        lat = float(line[0])
        lng = float(line[1])
        t = float(line[2])
        gps_data.append([lat, lng, t])
    return gps_data


# 根据 经纬度 获得网格编码 返回 0 <= x, y < 256
def get_id(lat, lng):
    step = (maxLat - minLat) / cell
    x = int((lat - minLat) / step)
    step = (maxLng - minLng) / cell
    y = int((lng - minLng) / step)
    return x, y


# 停留点检测
def get_stay_point(tr):   # tr = {[lat, lng, t], ...}
    length = len(tr)
    SP = []

    i = 0
    while i < length:

        j = i + 1
        while j < length:
            dis = geod.Inverse(tr[i][0], tr[i][1], tr[j][0], tr[j][1])
            if dis['s12'] > 300:    # 周围300米
                delt_t = tr[j][2] - tr[i][2]
                if delt_t > 600:    # 10分钟
                    lat = 0
                    lng = 0
                    for k in range(i, j):
                        lat = lat + tr[k][0]
                        lng = lng + tr[k][1]
                    lat = lat / (j - i)
                    lng = lng / (j - i)
                    st = tr[j][2] - tr[i][2]
                    SP.append([round(lat, 6), round(lng, 6), st])
                    i = j
                    break
            j = j + 1

        i = i + 1
    return SP
