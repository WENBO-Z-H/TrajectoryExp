"""
使用TSHN算法辨别真实轨迹和虚拟轨迹
"""

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.nn import functional as F
import MNAlgorithm
import DataPreparation


class TrajectoryDataset(Dataset):
    """
    依据Pytorch的规则制作数据集
    """
    def __init__(self, train_mode):
        """
        数据初始化
        :param train_mode: 训练模式和测试模式返回不同的数据集
        """
        self.train_mode = train_mode
        # 首先准备真实轨迹数据并编码
        data = DataPreparation.GetData("000")  # 该函数设定是读取100条数据
        real_traj_feature_list = []
        for i in range(len(data)):
            M = DataPreparation.EncodeTrajWithIT2I([data[0]])
            Ind = DataPreparation.EncodeInd(data[0])
            real_traj_feature_list.append([M, Ind])
        # 然后准备虚拟轨迹数据并编码
        dummy_traj_feature_list = self.GenerateDummyTrajectory(100)
        #print(real_traj_feature_list)
        # print(len(real_traj_feature_list))
        #print(dummy_traj_feature_list)

        print("数据读取和生成完毕")
        # 格式转换，划分训练集和测试集
        training_set = []
        testing_set = []
        for i in range(79):  # 79组训练集，每个数据是一个元组，第一个元素是M的差值，第二个元素是Ind的差值
            training_set.append([real_traj_feature_list[i][0] - real_traj_feature_list[i + 1][0],
                                 real_traj_feature_list[i][1] - real_traj_feature_list[i + 1][1]])  # 79组真真数据
        for i in range(79):  # 79组训练集
            training_set.append([real_traj_feature_list[i][0] - dummy_traj_feature_list[i + 1][0],
                                 real_traj_feature_list[i][1] - dummy_traj_feature_list[i + 1][1]])  # 79组真假数据

        for i in range(80, 99):  # 19组测试集
            testing_set.append([real_traj_feature_list[i][0] - real_traj_feature_list[i + 1][0],
                                real_traj_feature_list[i][1] - real_traj_feature_list[i + 1][1]])  # 19组真真数据
        for i in range(80, 99):  # 19组测试集
            testing_set.append([real_traj_feature_list[i][0] - dummy_traj_feature_list[i + 1][0],
                                real_traj_feature_list[i][1] - dummy_traj_feature_list[i + 1][1]])  # 19组真假数据

        if self.train_mode:
            self.data = training_set
        else:
            self.data = testing_set
        # print(self.data)
        # print(len(self.data))

    def __getitem__(self, index):
        """
        必须实现，作用是:获取索引对应位置的一条数据
        :param index: 索引号
        :return: data, label
        """
        if self.train_mode:  # 训练集
            if index < 79:  # 对应真真数据
                return self.data[index], 100.
            else:  # 对应真假数据
                return self.data[index], 0.
        else:  # 测试集
            if index < 19:  # 对应真真数据
                return self.data[index], 100.
            else:  # 对应真假数据
                return self.data[index], 0.

    def __len__(self):
        """
        必须实现，作用是得到数据集的大小
        :return: 数据集大小
        """
        if self.train_mode:
            return 158
        else:
            return 38

    def GenerateDummyTrajectory(self, n):
        """
        生成多条符合一定格式的虚拟轨迹
        :param n: 生成虚拟轨迹的条数
        :return: 将结果保存至self.dummy_data
        """
        trajetorys_feature = []
        for i in range(n):  # 生成x条轨迹
            #print(i)
            dummy_traj = MNAlgorithm.MovingInNeighborhood(0.0001, 100, with_t=True)  # 生成单条轨迹
            # 单条轨迹处理
            M = DataPreparation.EncodeTrajWithIT2I([dummy_traj])
            Ind = DataPreparation.EncodeInd(dummy_traj)
            #print(dummy_traj)
            trajetorys_feature.append([M, Ind])
        return trajetorys_feature


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
        #print(x)
        #print(y.shape)
        # 1. 对5D矩阵的卷积处理和打平部分
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        x = F.relu(x)
        #print(x)

        # 2. 对5维向量的全连接层部分
        y = self.fc3(y)
        y = F.relu(y)
        #print(x)
        y = self.fc4(y)
        y = F.relu(y)

        # 3. 1和2的结果向量合并部分
        xy = torch.cat((x, y), dim=1)
        xy = self.combine(xy)
        xy = F.relu(xy)
        output = self.fc5(xy)
        return output.squeeze(-1)


def CalAccuracy(pred, label):
    print(pred)
    print("---")
    print(label)
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i, v in enumerate(pred):
        # val = abs(v[i].item() - label[i].item())
        val = v[i].item()
        if val >= 50 and label[i].item() > 99:  # 预测为真
            tp = tp + 1
        if val < 50 and label[i].item() < 1:
            tn = tn + 1
        if val >= 50 and label[i].item() < 1:
            fn = fn + 1
        if val < 50 and label[i].item() > 99:
            fp = fp + 1

    return tp, tn, fn, fp


def TrainAndEvaluate(train_loader, test_loader):
    num_epochs = 50

    model = TSHN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        tmp_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            output = model(data[0], data[1])
            target = target.to(torch.float32)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp_loss = loss.data.numpy()

        print("epoch:", epoch)
        print("loss:", tmp_loss)

        #exit(0)

        TP = 0  # 被模型预测为正类的正样本
        TN = 0  # 被模型预测为负类的负样本
        FN = 0  # 被模型预测为正类的负样本
        FP = 0  # 被模型预测为负类的正样本
        model.eval()
        for (data_test, target_test) in test_loader:
            output_test = model(data_test[0], data_test[1])
            target_test = target_test.to(torch.float32)
            tp, tn, fn, fp = CalAccuracy(output_test, target_test)
            TP = TP + tp
            TN = TN + tn
            FN = FN + fn
            FP = FP + fp
        acc = (TP + TN) / (TP + TN + FN + FP)
        print("acc:", acc, "TP:", TP, "TN:", TN, "FN:", FN, "FP:", FP)


if __name__ == '__main__':
    training_set = TrajectoryDataset(train_mode=True)
    testing_set = TrajectoryDataset(train_mode=False)
    # exit(0)
    batch_size = 5
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True, num_workers=0)
    TrainAndEvaluate(train_loader, test_loader)
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print(data[0])
    #     print(data[0].shape)
    #     #print(data[1])
    #     exit(0)

