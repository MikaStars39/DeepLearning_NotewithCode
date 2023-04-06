import torch
import numpy

x_data = torch.randn(5, 1)
x_data = x_data
print(x_data)
y_data = torch.tensor([[10.0], [10.0], [10.0], [10.0], [10.0]])
print(y_data)


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # super()可以访问子类的父类，因为python3可以自动确定父类和自己是谁，所以LinearModule, self可以不写
        # 但是这里写上的话代码可读性更强，我是这么理解的
        self.linear = torch.nn.Linear(1, 1)
        # 构造一个对象，1个输入一个输出

    def forward(self, x):
        y_pred = self.linear(x)
        # 计算 y^ = (W^t)*X + b
        return y_pred


model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)
# 损失率的计算器，不除以样本的个数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 优化器

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
# output w and b

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)


