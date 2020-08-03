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
import datetime

from torch.utils.tensorboard import SummaryWriter
from Net import Network
from sklearn.metrics import confusion_matrix

from RunBuilder import RunBuilder
from RunManager import RunManager

from resources.plot_confusion_matrix import plot_confusion_matrix


# 读取数据集
train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


# 创建一个词典，并通过遍历获取三个参数的所有组合，能避免多个训练的嵌套
parameters = dict(
    lr=[.001],
    batch_size=[100, 1000],
    # shuffle=[True, False],
    device=['cuda', 'cpu']
)

m = RunManager()
for run in RunBuilder.get_runs(parameters):

    device = torch.device(run.device)
    network = Network().to(device)

    loader = DATA.DataLoader(train_set, batch_size=run.batch_size)
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
m.save('save_results')

# 训练完后将参数保存，以便下次能够直接加载模型
# torch.save(network.state_dict(), './data/net_params.pkl')

