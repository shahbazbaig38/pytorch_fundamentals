import torch

# Create a tensor
tensor_1 = torch.tensor([[1, 2, 3], [3, 4, 5]])

tensor_2 = torch.tensor([[6, 7, 8], [9, 10, 11]])

# multiply tensors (matrix multiplication)
print(tensor_1.matmul(tensor_2.T))
print(tensor_1 @ tensor_2.T)
print(torch.matmul(tensor_1, tensor_2.T))