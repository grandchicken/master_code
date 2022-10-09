import torch

a = torch.zeros((2,4,10))
b = torch.zeros((3,4,10))
c = torch.concat((a,b),dim=0)
print(c.shape)