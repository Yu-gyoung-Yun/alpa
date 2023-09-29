import torch

t = torch.tensor([1]).cuda(1)
t.to(2)
