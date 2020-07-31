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

# 传输一张图片至已经定义好的网络中并获取输出（尚未定义反向传播）
sample = next(iter(train_set))
image, label = sample
print("单张图片的shape", image.shape)
# 查看该图片的内容
plt.imshow(image.squeeze(0), cmap="gray")
plt.show()
# 传入需为rank=4，要有batch，将单张图片展到4维
image = image.unsqueeze(0)
print("unsqueeze后的单张图片shape", image.shape)
# 预测输出,打印实际label与预测的label
pred = network(image)
print("实际标签", label, "   预测标签：", pred.argmax(dim=1))

# 传输一个batch的图片进入网络并获取输出
# 获取一个批次的十张图片，并将其保存在images与labels两个tensor中
batch = next(iter(train_loader))
images, labels = batch
print("images shape", images.shape, "labels shape", labels.shape)
# 获取一个batch图片的预测输出10*10
preds = network(images)
print("preds shape", preds.shape)
print(preds.argmax(dim=1))
print(preds)
print(labels)
# 获取预测正确的那副图的个数
print(preds.argmax(dim=1).eq(labels).sum().item())