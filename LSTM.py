"""
使用LSTM算法辨别真实轨迹和虚拟轨迹
"""

import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import MNAlgorithm

# 超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
len_sections = 7  # 每个轨迹段的轨迹点数
input_size = 2  # 神经网络每个输入的大小，2表示一个经度一个纬度
hidden_size = 10  # 隐藏层单元数
num_layers = 1  # LSTM网络层数
num_classes = 2  # 类别数，只有真轨迹和假轨迹两类
batch_size = 128  # 批处理的每批数据量
num_epochs = 1  # 训练轮数
learning_rate = 0.001  # 学习率
threshold = 0.5  # 判断一个轨迹真假的阈值，若真轨迹段的比例大于阈值则认为是真轨迹


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


class RNN(nn.Module):
    """
    虚拟轨迹识别的LSTM方法
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # 初始化隐藏层和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # 此处的-1说明我们只取RNN最后输出的那个hn
        out = self.softmax(out)
        return out


def Train(train_loader):
    """
    模型训练
    :param train_loader: 符合pytorch规则的训练集DataLoader
    :return: 训练好的模型
    """
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("开始训练")

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (sections, labels) in enumerate(train_loader):
            sections = sections.reshape(-1, len_sections, input_size).to(device)
            labels = labels.to(device)

            outputs = model(sections)
            print(outputs)
            print(outputs.shape)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    return model


def Test(model, test_loader):
    """
    模型测试
    :param model: 训练好的模型
    :param test_loader: 符合pytorch规则的训练集DataLoader
    :return: 把模型保存到本地文件model.ckpt
    """
    # 开始测试
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sections, labels in test_loader:
            sections = sections.reshape(-1, len_sections, input_size).to(device)
            labels = labels.to(device)
            outputs = model(sections)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test sections: {} %'.format(100 * correct / total))

    # 保存模型检查点
    torch.save(model.state_dict(), 'model.ckpt')


# Recognition
def Recognition(model, traj):
    """
    检测一个轨迹是真轨迹还是虚拟轨迹
    :param model: 训练好的模型
    :param traj: tensor格式的轨迹，维度为[x, 2], x为轨迹长度
    :return:
    """
    traj = traj[:(len(traj) // len_sections) * len_sections]  # 切除分段多余部分
    traj = np.array(traj).reshape(len(traj) // len_sections, len_sections, 2)  # 分段
    traj = torch.from_numpy(traj).float()
    num_real = 0  # 真轨迹段个数
    num_total = traj.shape[0]  # 总轨迹段个数

    for i in num_total:
        section = traj[i]  # 截取出轨迹段
        result = model(section)  # 对轨迹段使用LSTM鉴别
        _, predicted = torch.max(result.data, 1)
        if predicted == 1:
            num_real += 1
    if num_real / num_total > threshold:
        return True


if __name__ == '__main__':
    # 数据集准备
    dataset = TrajectoryDataset(len_sections)
    train_size, test_size = int(0.8 * len(dataset)), int(0.2 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # 开始训练
    model = Train(train_loader)
    # 开始测试
    Test(model, test_loader)

    #traj = torch.randn(())
    #Recognition(model, traj)
