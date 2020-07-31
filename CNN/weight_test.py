import torch

from Net import Network

network = Network()
# 打印网络结构与每个层的权重矩阵的size，网络的toString从nn.Module中继承
print(network)
for name, param in network.named_parameters():
    print(name, '\t\t', param.size())
# 测试将网络中的权重矩阵用自定义的权重矩阵替代（为简化将network中的fc1的大小改为4,3）
fc1 = network.fc1
fc1_in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
fc1_weight_matrix = torch.tensor([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6]
], dtype=torch.float32)
print(fc1_weight_matrix.matmul(fc1_in_features))
# torch随机初始化权重矩阵时的输出
# 能够给这样调用fc1()作为一个方法的原因是Module实现了__call__方法
print(fc1(fc1_in_features))
# 我们制定初始权重矩阵并输出,因为有bias所以不是标准的30,40,50，若定义层时将bias设为false，则为标准输出30,40,50
network.fc1.weight = nn.Parameter(fc1_weight_matrix)
print(fc1(fc1_in_features))