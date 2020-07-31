import torch

t1 = torch.tensor([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
])
t2 = torch.tensor([
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
])
t3 = torch.tensor([
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
])
t = torch.stack((t1, t2, t3))
print(t.shape)
t = t.reshape(3, 1, 4, 4)
print(t[0])
print(t[0][0])
print(t[0][0])
print(t[0][0][0])
print(t[0][0][0][0])

t = t.flatten(start_dim=1)
print(t)