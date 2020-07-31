import torchvision
import torch.utils.data as DATA
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from Net import Network

tb = SummaryWriter()


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

network = Network()
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

tb.add_image('images', grid)
tb.add_graph(network, images)
tb.close()
