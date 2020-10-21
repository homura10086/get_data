from abc import ABC
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import init
from tool import *

torch.manual_seed(1)  # reproducible
torch.cuda.manual_seed_all(1)  # 为所有GPU设置随机种子

# Hyper Parameters
B_INIT = -0.2
num_RAN = int(1e4)
num_cell = 6
num_sample = (18 // num_cell) * num_RAN
num_feature = 5
rate_test = 0.2
batch_size = 256
lr = 0.01
num_epochs = 50
device = 'cuda'

# 数据处理
feature = pd.read_csv('data.csv', header=0, usecols=range(num_feature))
label = pd.read_csv('data.csv', header=0, usecols=[num_feature])
feature_normalize = np.zeros((num_sample * num_cell, num_feature))
for i in range(num_feature):
    operation_feature = np.array(feature[feature.columns[i]])
    feature_normalize[:, i] = minmaxscaler(operation_feature)
features = torch.from_numpy(feature_normalize).reshape(num_sample, 1, num_cell, num_feature)
label = torch.Tensor(label.values).squeeze()
labels = torch.zeros(num_sample, dtype=torch.int64)
for i in range(0, num_sample * num_cell, 6):
    labels_temp = label[i:i+6]
    flag = 0
    for lab in labels_temp:
        if lab != 0:
            labels[i // num_cell] = lab
            flag = 1
            break
    if flag == 0:
        labels[i // num_cell] = 0
# print("0:", tuple(labels).count(0), '\n' "1:", tuple(labels).count(1), '\n' "2:", tuple(labels).count(2),
# '\n' "3:", tuple(labels).count(3), '\n' "total:", tuple(labels).count(0) + tuple(labels).count(1) + tuple(
# labels).count(2) + tuple(labels).count(3))

# 数据集处理
dataset = Data.TensorDataset(features, labels)
num_test = int(rate_test * num_sample)
num_train = num_sample - num_test
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test], generator=None)
# print("train_dataset: ", len(train_dataset), '\n' ''"test_dataset: ", len(test_dataset))
train_iter = Data.DataLoader(
    dataset=train_dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=0,  # 多线程来读数据
    pin_memory=True
)
test_iter = Data.DataLoader(
    dataset=test_dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=0,  # 多线程来读数据
    pin_memory=True
)


def _set_init(layer):
    init.normal_(layer.weight, mean=0., std=.1)
    init.constant_(layer.bias, B_INIT)


class CNN(nn.Module, ABC):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 6, 5)
            # nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1,  # input height
                      out_channels=5,  # n_filters
                      kernel_size=(3, 2),  # filter size
                      stride=1,  # filter movement/step
                      padding=0,  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2
                      # if stride=1
                      ),  # output shape (5, 4, 4)
            # nn.Dropout(0.5),
            # nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            # nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 2, 4)
        )
        self.conv2 = nn.Sequential(  # input shape (5, 4, 4)
            nn.Conv2d(in_channels=5,
                      out_channels=10,
                      kernel_size=1,
                      stride=1,
                      padding=0),  # output shape (10, 4, 4)
            # nn.Dropout(0.5),
            # nn.BatchNorm2d(32),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # output shape (10, 2, 2)
        )
        self.out = nn.Sequential(
            nn.Linear(10 * 2 * 2, 4),  # fully connected layer, output 5 classes
            # nn.Softmax(dim=0),
        )
        # _set_init(self.out)  # initialization

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 10 * 2 * 2)
        output = self.out(x)
        return output


if __name__ == '__main__':
    cnn = CNN()
    # print(cnn)  # net architecture
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)  # optimize all cnn parameters
    train_ch5(cnn, train_iter, test_iter, optimizer, device, num_epochs)
    # torch.save(cnn, 'cnn.pkl')  # save entire net
    torch.save(cnn.state_dict(), 'cnn_params.pkl')  # save only the parameters
    
    # cnn.load_state_dict(torch.load('./cnn_params.pkl'))
    # data_iter = Data.DataLoader(
    #     dataset=dataset,  # torch TensorDataset format
    #     batch_size=batch_size,  # mini batch size
    #     shuffle=True,  # 要不要打乱数据 (打乱比较好)
    #     num_workers=0,  # 多线程来读数据
    #     pin_memory=True,
    #     drop_last=True
    # )
    # softmax(data_iter, device, cnn, batch_size)
