from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as DATA
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

from Net import Network

from RunBuilder import RunBuilder
from RunManager import RunManager



# 读取数据集
train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# 创建连个network，一个有batch normalization一个没有
torch.manual_seed(50)
network1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.Flatten(start_dim=1)
    , nn.Linear(in_features=12*4*4, out_features=120)
    , nn.ReLU()
    , nn.Linear(in_features=120, out_features=60)
    , nn.ReLU()
    , nn.Linear(in_features=60, out_features=10)
)

torch.manual_seed(50)
network2 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.BatchNorm2d(6)  # 输入应该是上面输出的channel数
    , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.Flatten(start_dim=1)
    , nn.Linear(in_features=12*4*4, out_features=120)
    , nn.ReLU()
    , nn.BatchNorm1d(120)
    , nn.Linear(in_features=120, out_features=60)
    , nn.ReLU()
    , nn.Linear(in_features=60, out_features=10)
)

networks = {
    'no_batch_norm': network1,
    'batch_norm': network2
}

# 创建一个词典，并通过遍历获取三个参数的所有组合，能避免多个训练的嵌套
parameters = dict(
    lr=[.001],
    batch_size=[1000],
    shuffle=[True],
    device=['cuda'],
    network=list(networks.keys())

)

m = RunManager()
for run in RunBuilder.get_runs(parameters):

    device = torch.device(run.device)
    network = networks[run.network].to(device)

    loader = DATA.DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, loader)
    for epoch in range(2):
        m.begin_epoch()
        for batch in loader:
            images = batch[0].to(device)  # 将传入网络的tensor也改为cuda
            labels = batch[1].to(device)
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()  # 初始化梯度
            loss.backward()  # 计算权重
            optimizer.step()  # 更新权重

            m.track_loss(loss)
            m.track_num_correct(preds, labels)

        m.end_epoch()
    m.end_run()
# m.save('save_results')

# 训练完后将参数保存，以便下次能够直接加载模型
# torch.save(network.state_dict(), './data/net_params.pkl')

