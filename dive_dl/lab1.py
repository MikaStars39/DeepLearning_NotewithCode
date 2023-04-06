import torch

x_data = torch.randn(10, 1, dtype=torch.float32)
y_data = x_data + 10


class TestLinearModel(torch.nn.Module):
    def __init__(self):
        super(TestLinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, X):
        y_pred = self.linear(X)
        return y_pred


model = TestLinearModel()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

