# some basic functions
import torch
import numpy as np

create_tensor1 = torch.ones(24)
create_tensor2 = torch.zeros(24)
create_tensor3 = torch.rand(24)
create_tensor4 = torch.randn(24, dtype=torch.float32)

view_tensor = create_tensor1.view(-1, 8)
# -1 means pytorch will auto-calculate
print(view_tensor.data)
# 好习惯 .data just read the data
# but if we directly visit it, it will build calculate graph

add_tensor = create_tensor1 + create_tensor2
add_tensor = add_tensor + torch.add(create_tensor1, create_tensor2)

print(add_tensor[1].item())
# single element uses item

num_array1 = np.ones(5)
torch_array1 = torch.from_numpy(num_array1)
num_array1 += 1
print(torch_array1)
# they use the same memory space

if torch.cuda.is_available():
    # check if cuda is available
    device_cuda = torch.device('cuda')
    tensor_device = torch.ones(5, device=device_cuda)
    tensor_device = tensor_device.to('cpu')
    # back to cpu

tensor_grad = torch.ones(5, requires_grad=True)
# default is false

with torch.no_grad():
    no_grad_tensor = torch.FloatTensor([1, 2, 3, 4])
    print(no_grad_tensor)
