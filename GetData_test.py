"""
读取Geolife数据集
"""
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
lat = []  # 纬度
lng = []  # 经度

path = "..\\Geolife Trajectories 1.3 - 副本" + "\\Data" + "\\000" + "\\Trajectory"  # 000的路径
plts_000 = os.scandir(path)

# 每一个文件的绝对路径（前100条）
i = 0
for item in plts_000:
    if i >= 100:
        break
    else:
        i += 1
    path_item = path + "\\" + item.name
    with open(path_item, 'r+') as fp:
        for item in fp.readlines()[6:]:
            item_list = item.split(',')
            lat.append(item_list[0])
            lng.append(item_list[1])

lat_new = [float(x) for x in lat]
lng_new = [float(x) for x in lng]
data = list(zip(lat_new, lng_new))

# 纬度范围
minLat = 39.7817
maxLat = 40.1075
# 经度范围
minLng = 116.1715
maxLng = 116.5728

for i in range(len(data)):
    if minLat <= data[i][0] <= maxLat and minLng <= data[i][1] <= maxLng:
        print(i, ": 符合")
