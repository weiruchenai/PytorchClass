import torch

t1 = torch.tensor([1, 1, 1, 1])
t2 = torch.tensor([2, 2, 2, 2])
t3 = torch.tensor([3, 3, 3, 3])
print(torch.cat((t1, t2, t3), dim=0))
print(torch.stack((t1, t2, t3), dim=0))
print(torch.stack((t1, t2, t3), dim=1))
print(torch.cat((t1.unsqueeze(1), t2.unsqueeze(1), t3.unsqueeze(1)), dim=0))
print(torch.cat((t1.unsqueeze(1), t2.unsqueeze(1), t3.unsqueeze(1)), dim=1))


# 三张images成为一个batch
image1 = torch.zeros(3, 28, 28)
image2 = torch.zeros(3, 28, 28)
image3 = torch.zeros(3, 28, 28)
print(torch.stack((image1, image2, image3), dim=0).shape)

# 三张batch size为1的images放到同一个batch中
image1 = torch.zeros(1, 3, 28, 28)
image2 = torch.zeros(1, 3, 28, 28)
image3 = torch.zeros(1, 3, 28, 28)
print(torch.cat((image1, image2, image3), dim=0).shape)

# 三张独立图片加到一个现有的batch中
image1 = torch.zeros(3, 28, 28)
image2 = torch.zeros(3, 28, 28)
image3 = torch.zeros(3, 28, 28)
batch = torch.zeros(1, 3, 28, 28)
print(torch.cat((
    batch,
    torch.stack((image1, image2, image3), dim=0)
),
    dim=0
).shape)