# 用人话讲一波softmax，简单来说就是前面所有的全连接和激活层都照样来，用的都是BCE
# 然后在最后一层，我们不用那个sigmoid激活了，最后一层我们肯定有好多值，有正有负
# 我们把这些值都取一个e为底的指数，然后每个都除以所有的和（这个过程就是归一化）
# 这样每个都是一个0到1之间的概率，而且这些概率的和为1
# 这样我们又能对这最后一层做BCE了
# 这里实际上不是BCE，是NLLLoss，Negative Log Likelihood Loss
# 用独热编码转换的1-9这种数字，如果该项为0，其损失也为0
import torch
import numpy as np
from torchvision import transforms
from torchvision import  datasets
from torch.utils.data import dataset
from torch.utils.data import dataloader

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, ))])

train_dataset = datasets.MNIST(root='data/MNIST',
                               )
