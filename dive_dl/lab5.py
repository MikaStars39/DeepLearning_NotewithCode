# Logistics Regression
# 交叉熵：这是一种衡量两种分布之间差异的方法
# 比如你有好多个箱子，里面有各种各样的颜色的球
# 第一个箱子有0.5 的概率摸出红球(p)，第二个箱子有0.3的概率摸出红球，当然，它们摸出其他球(q)的概率也不同
# 如何衡量这种不同呢？信息熵的概念给出的解释是：
# 定义一个函数 H(p,q) = -sum(从1到n）(pi * log(qi))
# 为什么要取对数？GPT说：取对数将乘法转化为加法。不取对数数太长容易浮点数下溢。
# 两个变量就是BCE Binary Cross Entropy Loss
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.Tensor([[0], [0], [1], [1]])
# 请用Tensor, tensor只是继承


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()
# define criterion and optimizer
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# start to iterate
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    # forward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_test = torch.tensor([[3.5]])
y_test = model(x_test)
print('y_pred = ', y_test.data)


x = np.linspace(0, 10, 200)
x_tensor = torch.Tensor(x).view((200, 1))
y_tensor = model(x_tensor)
y = y_tensor.data.numpy()
plt.plot(x, y)
plt.grid()
plt.xlabel('Study Time')
plt.ylabel('Probability of Death')
plt.show()

