import torch.nn.functional as F
import torch

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2])
b = torch.tensor([0.0])

z = w1 * x1 + b
print("z = ", z)

a = torch.sigmoid(z)
print("a = ", a)

loss = F.binary_cross_entropy(a, y)
print("loss = ", loss)  