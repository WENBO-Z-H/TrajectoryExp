"""
读取Geolife数据集
"""
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
lat = []  # 维度
lng = []  # 经度
# 总路径
path = os.getcwd() + "\\Geolife Trajectories 1.3" + "\\Data" + "\\000" + "\\Trajectory"  # 000的路径
# print(os.listdir(os.getcwd()+"\\Geolife Trajectories 1.3"+"\\Data"+"\\003"+"\\Trajectory"
plts_001 = os.scandir(path)
# 每一个文件的绝对路径
for item in plts_001:
    path_item = path + "\\" + item.name
    with open(path_item, 'r+') as fp:
        for item in fp.readlines()[6:]:
            item_list = item.split(',')
            lat.append(item_list[0])
            lng.append(item_list[1])

lat_new = [float(x) for x in lat]
lng_new = [float(x) for x in lng]
plt.ylim((min(lat_new), max(lat_new)))
plt.xlim((min(lng_new), max(lng_new)))

print(lat)
print(lng)
plt.title("000轨迹测试")
plt.xlabel("经度")  # 定义x坐标轴名称
plt.ylabel("维度")  # 定义y坐标轴名称
plt.plot(lng_new, lat_new)  # 绘图
plt.show()  # 展示
