from random import randint, sample
from numpy import random
import numpy
# import torch
# from torch.nn.utils.clip_grad import clip_grad_norm_

# modes = ['a' for i in range(18)]
# for i, x in enumerate(modes):
#     print(i, x)
# rand_index = random.randint(0, 5, randint(0, 6 - 1))
# rand_index = random.choice(a=range(6), replace=False, size=6)
# rand_indexs = sample(range(6), randint(1, 6))
# print(rand_indexs)

# n = 5  #类别数
# indices = torch.randint(0, n, size=(15,15))  #生成数组元素0~5的二维数组（15*15）
# one_hot = torch.nn.functional.one_hot(indices, n)  #size=(15, 15, n)
# print(indices,'\n',one_hot)
x = numpy.zeros(10)
y = numpy.ones(10)
for a, b in zip(x, y):
    print(a, b)
