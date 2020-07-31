import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as DATA
import matplotlib.pyplot as plt
import torch.optim as optim

from Net import Network

train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
train_loader = DATA.DataLoader(
    train_set,
    batch_size=100,
    shuffle=True
)
network = Network()
batch = next(iter(train_loader))  # 获取一个批次
images, labels = batch
# 调用Adam优化器，超参为学习率lr
optimizer = optim.Adam(network.parameters(), lr=0.01)
preds = network(images)  # 将这个batch传入网络
print(preds.shape, labels.shape)
loss = F.cross_entropy(preds, labels)  # 计算损失

loss.backward()  # 计算梯度
optimizer.step()  # 更新权重

print('loss1:', loss.item())
preds = network(images)
loss = F.cross_entropy(preds, labels)
print('loss2:', loss.item())
