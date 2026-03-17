import torch


# 0D tensor
tensor_0d = torch.tensor(5)
print(f"{tensor_0d} is a {tensor_0d.ndim}D tensor having shape {tensor_0d.shape} and data type {tensor_0d.dtype}")

# 1D tensor
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(f"{tensor_1d} is a {tensor_1d.ndim}D tensor having shape {tensor_1d.shape} and data type {tensor_1d.dtype}")

# 2D tensor
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"{tensor_2d} is a {tensor_2d.ndim}D tensor having shape {tensor_2d.shape} and data type {tensor_2d.dtype}")

# 3D tensor
tensor_3d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(f"{tensor_3d} is a {tensor_3d.ndim}D tensor having shape {tensor_3d.shape} and data type {tensor_3d.dtype}")

# float tensor
tensor_float32 = torch.tensor([1.0, 2.0, 3.0])
print(f"{tensor_float32} is a {tensor_float32.ndim}D tensor having shape {tensor_float32.shape} and data type {tensor_float32.dtype}")

# int to float tensor
int_to_float_tensor = tensor_0d.to(torch.float32)
print(f"{int_to_float_tensor} is a {int_to_float_tensor.ndim}D tensor having shape {int_to_float_tensor.shape} and data type {int_to_float_tensor.dtype}")