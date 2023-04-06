import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# DataLoader 和 dataloader 是等价的，表示的都是用于加载数据的迭代器。
import pandas as pd

# prepare dateset
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, ))])
train_dataset = datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transform)
train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transform)
test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)


class CnnModel(torch.nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size_f = x.size(0)
        # 如果要获取某个特定维度的大小，可以使用 size(dim) 函数来获取
        x = torch.nn.ReLU(self.conv1(x))
        x = torch.nn.ReLU(self.conv2(x))
        x = x.view(batch_size_f, -1)
        x = self.fc(x)
        return x


model = CnnModel()
device = torch.device("mps")
# model.to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_dataloader, 0):
        # prepare data
        inputs, label = data
        # forward
        optimizer.zero_grad()
        # 推荐在每个batch训练开始时调用optimizer.zero_grad()函数，清零梯度
        outputs = model(inputs)
        loss = criterion(outputs, label)
        running_loss += loss.item()
        # backward
        loss.backward()
        # update
        optimizer.step()

        if batch_idx % 300 == 299:
            print(f"[{epoch+1},{batch_idx+1}] loss: {running_loss/300}")
            running_loss = 0.0

def test():
    
