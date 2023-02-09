# 把一个轨迹分为n个轨迹段，检测后得到n个结果
import torch
import torch.nn as nn

# 输入数据 x 的向量维数 10, 设定 LSTM 隐藏层的特征维度 20, 此 model 用 2 个 LSTM 层
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)    # input(seq_len, batch, input_size)
h0 = torch.randn(2, 3, 20)       # h_0(num_layers * num_directions, batch, hidden_size)
c0 = torch.randn(2, 3, 20)       # c_0(num_layers * num_directions, batch, hidden_size)
# output(seq_len, batch, hidden_size * num_directions)
# h_n(num_layers * num_directions, batch, hidden_size)
# c_n(num_layers * num_directions, batch, hidden_size)
output, (hn, cn) = rnn(input, (h0, c0))

# torch.Size([5, 3, 20]) torch.Size([2, 3, 20]) torch.Size([2, 3, 20])
print(output.size(), hn.size(), cn.size())

# 输入数据 x 的向量维数 10, 设定 LSTM 隐藏层的特征维度 20, 此 model 用 2 个 LSTM 层
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)    # input(seq_len, batch, input_size)
h0 = torch.randn(2, 3, 20)       # h_0(num_layers * num_directions, batch, hidden_size)
c0 = torch.randn(2, 3, 20)       # c_0(num_layers * num_directions, batch, hidden_size)
# output(seq_len, batch, hidden_size * num_directions)
# h_n(num_layers * num_directions, batch, hidden_size)
# c_n(num_layers * num_directions, batch, hidden_size)
output, (hn, cn) = rnn(input, (h0, c0))

# torch.Size([5, 3, 20]) torch.Size([2, 3, 20]) torch.Size([2, 3, 20])
print(output.size(), hn.size(), cn.size())
