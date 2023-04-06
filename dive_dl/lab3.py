import matplotlib.pyplot as plt
import numpy as np
from d2l import torch
from torch.utils import data


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    # 生成大小为num_examples的X矩阵，其中每个元素在正态分布中抽样
    feature = np.random.normal(scale=1, size=(num_examples, len(w)))
    # 生成真实标签y
    label = np.dot(feature, w) + b  # 标签y为矩阵乘积Xw+b加上一个常数项
    # 添加正态分布的噪声到标签中
    label += np.random.normal(scale=0.01, size=label.shape)
    return feature, label


true_w = torch.tensor([2, -3, 4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    # 构造一个迭代器
    dataset = data.TensorDataset(*data_arrays)
    # 这里的*是为了解包，data_arrays是一个元组
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))
