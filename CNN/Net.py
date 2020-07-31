import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as DATA
import matplotlib.pyplot as plt
import torch.optim as optim


# 自己定义的网络必须继承nn.Module并实现其中的方法
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # 定义自己的网络，首先是两个卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        # 接下来是三个全连接层
        # self.fc1 = nn.Linear(in_features=4, out_features=3, bias=False)
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # input layer
        t = t
        # conv1 layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # conv2 layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # fc1 layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2 layer
        t = self.fc2(t)
        t = F.relu(t)

        # output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)
        return t




