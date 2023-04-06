import torch
import matplotlib.pyplot as plt
input_data = torch.randn(100, 5)
print(input_data)
x_data = input_data[:, 0]
y_data = input_data[:, 1]
plt.scatter(x_data, y_data)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
model = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

output = model(input_data)
print(output)
