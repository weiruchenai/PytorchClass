import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as DATA
import numpy as np
import torch.nn.functional as F

# 读取数据集
from torch import optim

from Net import Network
from RunBuilder import RunBuilder
from RunManager import RunManager

train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# easy way:一次性读取所有数据
loader = DATA.DataLoader(train_set, batch_size=len(train_set))
data = next(iter(loader))
print(data[0].mean(), data[0].std())

# hard way：数据集过大时采用，自己实现mean与std的计算
num_of_pixels = len(train_set) * 28 * 28
total_sum = 0
for batch in loader:
    total_sum += batch[0].sum()
mean = total_sum / num_of_pixels
sum_of_squared_error = 0
for batch in loader:
    # 所有批次的均方差加起来
    sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()
std = np.sqrt(sum_of_squared_error / num_of_pixels)
print(mean, std)

# 读取归一化数据集
train_set_normalization = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
)
# 重新计算mean与std
loader = DATA.DataLoader(train_set_normalization, batch_size=len(train_set))
data = next(iter(loader))
print(data[0].mean(), data[0].std())
# 创建一个词典，并通过遍历获取三个参数的所有组合，能避免多个训练的嵌套
train_sets = {
    'not_normal': train_set,
    'normal': train_set_normalization
}
parameters = dict(
    lr=[.001],
    batch_size=[100, 1000],
    # shuffle=[True, False],
    device=['cuda'],
    train_set=['not_normal', 'normal']
)

m = RunManager()
for run in RunBuilder.get_runs(parameters):

    device = torch.device(run.device)
    network = Network().to(device)

    loader = DATA.DataLoader(train_sets[run.train_set], batch_size=run.batch_size)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, loader)
    for epoch in range(10):
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
