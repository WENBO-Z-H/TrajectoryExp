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
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01


class TrajectoryDataset(Dataset):
    def __init__(self, len_sections):
        self.real_data = []
        self.dummy_data = []
        self.ReadGeoLifeDataset()  # 读取真实轨迹数据
        self.GenerateDummyTrajectory(10)  # 生成虚拟轨迹数据

        print("\n\n")
        print(self.real_data.shape)
        print(self.dummy_data.shape)

    def __getitem__(self, index):
        """
        必须实现，作用是:获取索引对应位置的一条数据
        :param index:
        :return:
        """
        return self.data[index]

    def __len__(self):
        """
        必须实现，作用是得到数据集的大小
        :return: 数据集大小
        """
        return len(self.data)

    def ReadGeoLifeDataset(self):
        """
        读取Geolife数据集
        :return: 将读取结果保存在self.real_data中
        """
        lat = []  # 维度
        lng = []  # 经度
        people_ids = os.scandir("..\\Geolife Trajectories 1.3" + "\\Data")
        for people_id in people_ids:
            print(people_id.name)
            # 数据集每个个体数据的总路径
            path = "..\\Geolife Trajectories 1.3" + "\\Data" + "\\" + people_id.name + "\\Trajectory"
            plts = os.scandir(path)
            # 每一个文件的绝对路径
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
        points = points[:(len(points) // len_sections) * len_sections]  # 切除分段多余部分
        self.real_data = np.array(points).reshape(len(points) // len_sections, len_sections, 2)  # 分段

    def GenerateDummyTrajectory(self, n):
        """
        生成多条符合一定格式的虚拟轨迹
        :param n: 生成虚拟轨迹的条数
        :return: 将结果保存至self.dummy_data
        """
        trajetorys = []
        for i in range(n):  # 生成x条轨迹
            dummy_traj = MNAlgorithm.MovingInNeighborhood(0.0001, 100)  # 生成单条轨迹
            # 单条轨迹处理
            dummy_traj = dummy_traj[:(len(dummy_traj) // len_sections) * len_sections]  # 切除分段多余部分
            arr = np.array(dummy_traj).reshape(len(dummy_traj) // len_sections, len_sections, 2)  # 分段（没有时间维度）
            trajetorys.append(arr)
        dummy_traj_arr = np.concatenate(trajetorys, axis=0)
        print(dummy_traj_arr[:2])
        self.dummy_data = dummy_traj_arr


class RNN(nn.Module):
    """
    Dummy Trajectory Recognition Scheme Based on LSTM (many-to-one)
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # 此处的-1说明我们只取RNN最后输出的那个hn
        return out


def train():
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (sections, labels) in enumerate(train_loader):
            sections = sections.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sections)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    return model


def Test(model):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sections, labels in test_loader:
            sections = sections.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(sections)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test sections: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')


# Recognition
def Recognition():
    pass


if __name__ == '__main__':
    """
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = train()
    Test(model)
    """
    len_sections = 5
    data = TrajectoryDataset(len_sections)
