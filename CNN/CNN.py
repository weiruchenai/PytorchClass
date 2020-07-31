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
# 数据集加载器
train_loader = DATA.DataLoader(
    train_set,
    batch_size=1000,
    shuffle=True
)


# 计算输出中正确的个数
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# 创建Network对象
network = Network()
# 创建优化器
optimizer = optim.Adam(network.parameters(), lr=0.01)
# 传入这个batch的图片
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)
tb = SummaryWriter()
tb.add_image('images', grid)
tb.add_graph(network, images)

# 创建循环，10个epoch，每个epoch更新600（60000/100）次权重
for epoch in range(10):
    start_time = datetime.datetime.now()
    total_loss = 0
    total_correct = 0
    # 获取一个batch100张图片
    for batch in train_loader:
        images, labels = batch
        preds = network(images)
        loss = F.cross_entropy(preds, labels)  # 计算损失
        # 先把梯度归零，并计算梯度以及更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    end_time = datetime.datetime.now()

    # tensorboard绘制图表
    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)
    tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

    tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
    tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
    tb.add_histogram(
        'conv1.weight.grad'
        , network.conv1.weight.grad
        , epoch
    )
    print("epoch:", epoch,
          "total loss:", total_loss,
          "total correct:", total_correct,
          "epoch time:", (end_time - start_time).seconds)

# 训练完后将参数保存，以便下次能够直接加载模型
# torch.save(network.state_dict(), './data/net_params.pkl')

