"""
使用TSHN算法辨别真实轨迹和虚拟轨迹
"""

import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F
import MNAlgorithm


class TrajectoryDataset(Dataset):
    """
    依据Pytorch的规则制作数据集
    """
    def __init__(self, len_sections):
        self.len_sections = len_sections
        self.real_data = []
        self.dummy_data = []
        self.ReadGeoLifeDataset()  # 读取真实轨迹数据
        self.GenerateDummyTrajectory(10000)  # 生成虚拟轨迹数据
        array_data = np.concatenate((self.real_data, self.dummy_data), axis=0)
        self.tensor_data = torch.from_numpy(array_data).float()

        print("\n\n")
        print(self.real_data.shape)
        print(self.dummy_data.shape)
        torch.set_printoptions(precision=16)
        print(self.tensor_data.shape)

    def __getitem__(self, index):
        """
        必须实现，作用是:获取索引对应位置的一条数据
        :param index: 索引号
        :return: data, label
        """
        self.true_index = self.real_data.shape[0]
        if index < self.true_index:
            label = 1
        else:
            label = 0
        return self.tensor_data[index], label

    def __len__(self):
        """
        必须实现，作用是得到数据集的大小
        :return: 数据集大小
        """
        return self.tensor_data.shape[0]

    def ReadGeoLifeDataset(self):
        """
        读取Geolife数据集
        :return: 将读取结果保存在self.real_data中
        """
        lat = []  # 维度
        lng = []  # 经度
        people_ids = os.scandir("..\\Geolife Trajectories 1.3 part" + "\\Data")
        for people_id in people_ids:
            print(people_id.name)
            # 数据集每个个体数据的总路径
            path = "..\\Geolife Trajectories 1.3 part" + "\\Data" + "\\" + people_id.name + "\\Trajectory"
            plts = os.scandir(path)
            # 每一个文件的路径
            for item in plts:
                path_item = path + "\\" + item.name
                with open(path_item, 'r+') as fp:
                    for item in fp.readlines()[6:]:
                        item_list = item.split(',')
                        lat.append(item_list[0])
                        lng.append(item_list[1])
            lat_new = [float(x) for x in lat]
            lng_new = [float(x) for x in lng]
        points = list(zip(lat_new, lng_new))
        points = points[:(len(points) // self.len_sections) * self.len_sections]  # 切除分段多余部分
        self.real_data = np.array(points).reshape(len(points) // self.len_sections, self.len_sections, 2)  # 分段

    def GenerateDummyTrajectory(self, n):
        """
        生成多条符合一定格式的虚拟轨迹
        :param n: 生成虚拟轨迹的条数
        :return: 将结果保存至self.dummy_data
        """
        trajetorys = []
        for i in range(n):  # 生成x条轨迹
            print(i)
            dummy_traj = MNAlgorithm.MovingInNeighborhood(0.0001, 100)  # 生成单条轨迹
            # 单条轨迹处理
            dummy_traj = dummy_traj[:(len(dummy_traj) // self.len_sections) * self.len_sections]  # 切除分段多余部分
            arr = np.array(dummy_traj).reshape(len(dummy_traj) // self.len_sections, self.len_sections, 2)  # 分段（没有时间维度）
            trajetorys.append(arr)
        dummy_traj_arr = np.concatenate(trajetorys, axis=0)
        self.dummy_data = dummy_traj_arr


class TSHN(nn.Module):
    def __init__(self):
        super(TSHN, self).__init__()
        # 1. 对5D矩阵的卷积处理和打平部分
        self.conv1 = nn.Sequential(     # (5, 256, 256)
            nn.Conv2d(5, 5, 2, 2, 0),   # (5, 128, 128)  params: in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(2)             # (5, 64, 64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 5, 2, 2, 0),   # (5, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2)             # (5, 16, 16)
        )
        self.fc1 = nn.Linear(5 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)

        # 2. 对5维向量的全连接层部分
        self.fc3 = nn.Linear(4, 16)
        self.fc4 = nn.Linear(16, 4)

        # 3. 1和2的结果向量合并部分
        self.combine = nn.Linear(132, 32)  # 128 + 4
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x, y):
        # 1. 对5D矩阵的卷积处理和打平部分
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        x = F.relu(x)
        print(x)

        # 2. 对5维向量的全连接层部分
        y = self.fc3(y)
        y = F.relu(y)
        print(x)
        y = self.fc4(y)
        y = F.relu(y)

        # 3. 1和2的结果向量合并部分
        xy = torch.cat((x, y), dim=1)
        xy = self.combine(xy)
        xy = F.relu(xy)
        output = self.fc5(xy)
        return output


if __name__ == '__main__':
    x = torch.randn((1, 5, 256, 256))
    model = TSHN()
    y = model(x)
    print(y)

