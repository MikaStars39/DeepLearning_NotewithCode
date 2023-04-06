import torch
B = torch.arange(9, dtype=torch.float32).reshape(3, 3)
A = B.clone()
print(A.sum(axis=[0, 1], keepdims=True)/A, A.mean(), A.numel(), A.cumsum(axis=0))

