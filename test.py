from random import randint, sample, choices
from numpy import random
import numpy
import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import pandas as pd

# modes = ['a' for i in range(18)]
# rand_index = random.randint(0, 5, randint(0, 6 - 1))
# rand_index = random.choice(a=range(6), replace=False, size=6)
# rand_indexs = sample(range(6), randint(1, 6))
# print(rand_indexs)

# n = 5  #类别数
# indices = torch.randint(0, n, size=(15,15))  #生成数组元素0~5的二维数组（15*15）
# one_hot = torch.nn.functional.one_hot(indices, n)  #size=(15, 15, n)
# print(indices,'\n',one_hot)

# for a, b in zip(x, y):
#     print(a, b)

# label = choices([i for i in range(5)], weights=(1, 3, 3, 3, 3), k=1)
# print(label[0], label[-1])

# x = torch.randint(low=1, high=3, size=(17, 3, 3))
# m = nn.Sequential(
#     nn.Flatten(start_dim=0, end_dim=-1)
# )
# print(m(x).size())

# data = pd.read_csv('data1.csv', usecols=range(18))
# data1 = data.iloc[:, :-1]
# x = pd.DataFrame()
# x = x.append(data1.iloc[1:10000])
# x = x.append(data1.iloc[10000:20000])
# print(x)

# dataset = data_process('origin')
# data_iter = Data.DataLoader(
#         dataset=dataset,  # torch TensorDataset format
#         batch_size=batch_size,  # mini batch size
#         shuffle=True,  # 要不要打乱数据 (打乱比较好)
#         num_workers=0,  # 多线程来读数据
#         pin_memory=True,
#         drop_last=True
#     )
# for X, y in data_iter:
#     cnn = CNN()
#     cnn = cnn.to(device)
#     print(cnn.device)
#     X = X.float().to(device)
#     y = y.to(device)
#     cnn.load_state_dict(torch.load('cnn_origin.pkl'))
#     print(cnn.device)
#     y_hat1 = cnn(X)
#     y_hat1_max = y_hat1.argmax(dim=1)

