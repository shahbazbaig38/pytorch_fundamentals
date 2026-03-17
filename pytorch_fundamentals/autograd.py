import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = w1 * x1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

print('a = ', a)
print('loss = ', loss)


# Compute gradients with respect to w1 and b
grad_w1 = grad(loss, w1, retain_graph=True)
grad_b = grad(loss, b, retain_graph=True)

print("Gradient with respect to w1: ", grad_w1)
print("Gradient with respect to b: ", grad_b)

# Alternatively, we can call backward() to compute gradients for all parameters
loss.backward()

print("Gradient with respect to w1 (using backward): ", w1.grad)
print("Gradient with respect to b (using backward): ", b.grad)