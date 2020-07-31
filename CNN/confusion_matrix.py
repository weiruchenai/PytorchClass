import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as DATA
import matplotlib.pyplot as plt
import torch.optim as optim


from Net import Network
from sklearn.metrics import confusion_matrix
from resources.plot_confusion_matrix import plot_confusion_matrix


# 构建模型对象
network = Network()
# 将模型参数加载到新模型中
state_dict = torch.load('./data/net_params.pkl')
network.load_state_dict(state_dict)

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


# 获取一个能返回所有6w个preds的方法
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels_1 = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            , dim=0
        )
    return all_preds


# 获取到包含所有predictions的tensor
with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

preds_correct = get_num_correct(train_preds, train_set.targets)
print('total correct:', preds_correct)
print('accuracy:', preds_correct / len(train_set))

# 将labels与预测结果堆叠成6w*2的tensor
stacked = torch.stack(
    (
        train_set.targets,
        train_preds.argmax(dim=1)),
    dim=1)
print(stacked.shape)
print(stacked)
# 初始化混淆矩阵为全0
cmt = torch.zeros(10, 10, dtype=torch.int64)
# 根据堆叠后的tensor，预测正确时，才会在对角线上+1
for p in stacked:
    true_label, pred_label = p.tolist()
    cmt[true_label, pred_label] = cmt[true_label, pred_label] + 1
print(cmt)

# 也可以用sklearn.metrics库中的方法来生成混淆矩阵
cmt2 = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
print(cmt2)
# 绘制混淆矩阵图，传入混淆矩阵与类别
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cmt2, train_set.classes)
