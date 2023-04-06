import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# Dataset is an abstract class
# DataLoader is used to load data


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        original_data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # 获取数据，用逗号分割，使用float32的原因是大部分GPU只支持这个形式，需要用到GPU加速计算
        self.len = original_data.shape[0]
        # Here .shape will return a tuple (a, b, ...), a is the number of first dimension
        self.x_data = torch.from_numpy(original_data[:, :-1])
        self.y_data = torch.from_numpy(original_data[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('data/diabetes.csv')
train_loader = DataLoader(dataset=dataset,  # 数据集
                          batch_size=32,  # 训练集的数量
                          shuffle=True,  # 是否打乱
                          num_workers=0)  # 线程数量


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


if __name__ == '__main__':
    for epoch in range(1000):
        # 一个epoch是所有数据跑一遍
        for i, data in enumerate(train_loader, 0):
            # 1. prepare data
            inputs, labels = data
            # 2. forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # 3. backward
            optimizer.zero_grad()
            loss.backward()
            # 4. update
            optimizer.step()



