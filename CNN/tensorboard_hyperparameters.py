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

from itertools import product
from torch.utils.tensorboard import SummaryWriter
from Net import Network


# 读取数据集
train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


# 计算输出中正确的个数
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# 创建一个词典，并通过遍历获取三个参数的所有组合，能避免多个训练的嵌套
parameters = dict(
    lr=[.01, .001],
    batch_size=[10, 100, 1000],
)
param_values = [v for v in parameters.values()]
print(param_values)
# 查看所有参数的组合
for lr, batch_size in product(*param_values):
    print(lr, batch_size)


# 传入所有参数组合
for lr, batch_size in product(*param_values):
    # 创建Network对象
    network = Network()
    # 创建优化器
    optimizer = optim.Adam(network.parameters(), lr=lr)
    # 数据集加载器
    train_loader = DATA.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)

    comment = f' batch_size={batch_size} lr={lr}'
    tb = SummaryWriter(comment=comment)
    tb.add_image('images', grid)
    tb.add_graph(network, images)

    # 创建循环，10个epoch，每个epoch更新600（60000/100）次权重
    for epoch in range(2):
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
            # 计算损失与正确个数
            total_loss += loss.item() * batch_size
            total_correct += get_num_correct(preds, labels)

        end_time = datetime.datetime.now()

        # tensorboard绘制图表
        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Number Correct', total_correct, epoch)
        tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

        # tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
        # tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
        # tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, epoch)
        # 用for循环替代上面的几行代码
        for name, weight in network.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(
                f'{name}.grad'
                , weight.grad
                , epoch
            )
        print("epoch:", epoch,
              "batch size:", batch_size,
              "Adam learning rate:", lr,
              "total loss:", total_loss,
              "total correct:", total_correct,
              "accuracy:", total_correct / len(train_set),
              "epoch time:", (end_time - start_time).seconds)
    tb.close()
# 训练完后将参数保存，以便下次能够直接加载模型
# torch.save(network.state_dict(), './data/net_params.pkl')

