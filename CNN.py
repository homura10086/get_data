import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import init
from main import num_core
from tool import *
import torch

torch.manual_seed(1)  # reproducible
torch.cuda.manual_seed_all(1)  # 为所有GPU设置随机种子

# Hyper Parameters
# B_INIT = -0.2
num_cell = 18
num_feature = 17
rate_test = 0.2
batch_size = 256
lr = 0.001
num_epochs = 5
device = 'cuda'


def data_process(data_name):
    # 数据处理

    # 合并并保存数据
    # data_list = []
    # for i in range(num_core):
    #     data_temp = pd.read_csv('data' + str(i + 1) + '.csv', header=0, usecols=range(num_feature + 1))
    #     data_list.append(data_temp)
    # data = pd.concat(data_list)
    # data.to_csv('data_origin.csv')

    if data_name == 'origin' or data_name == 'test':
        data = pd.read_csv('data_origin.csv', header=0, usecols=range(1, num_feature+2))
    elif data_name == 'coverage':
        data = pd.read_csv('data1.csv', header=0, usecols=range(num_feature+1))
    elif data_name == 'capacity':
        data = pd.read_csv('data2.csv', header=0, usecols=range(num_feature+1))
    else:
        data = pd.read_csv('data3.csv', header=0, usecols=range(num_feature+1))
    feature = data.iloc[:, :-1]
    label = data.iloc[:, -1]
    # 标签处理
    num_sample = label.shape[0] // num_cell
    label = torch.Tensor(label.values).squeeze()
    labels = torch.zeros((num_sample, 2), dtype=torch.int64)
    for i in range(0, num_sample * num_cell, num_cell):
        labels_temp = label[i:i + num_cell]
        for lab in labels_temp:
            if lab in range(1, 5):
                labels[i // num_cell][0] = 1
                labels[i // num_cell][1] = lab
                break
            elif lab in range(5, 8):
                labels[i // num_cell][0] = 2
                if data_name == 'test':
                    labels[i // num_cell][1] = lab
                else:
                    labels[i // num_cell][1] = lab - 4
                break
            elif lab in range(8, 11):
                labels[i // num_cell][0] = 3
                if data_name == 'test':
                    labels[i // num_cell][1] = lab
                else:
                    labels[i // num_cell][1] = lab - 7
                break
    # 特征归一化
    feature_normalize = np.zeros((num_sample * num_cell, num_feature))
    for i in range(num_feature):
        operation_feature = np.array(feature[feature.columns[i]])
        feature_normalize[:, i] = minmaxscaler(operation_feature)
    features = torch.from_numpy(feature_normalize).reshape(num_sample, 1, num_cell, num_feature)

    # for test
    # print("0:", tuple(labels[:, 1]).count(0), '\n' "1:", tuple(labels[:, 1]).count(1), '\n' "2:",
    #       tuple(labels[:, 1]).count(2), '\n' "3:", tuple(labels[:, 1]).count(3), '\n' "4:",
    #       tuple(labels[:, 1]).count(4), '\n' "5:", tuple(labels[:, 1]).count(5), '\n' "6:",
    #       tuple(labels[:, 1]).count(6), '\n' "7:", tuple(labels[:, 1]).count(7), '\n' "8:",
    #       tuple(labels[:, 1]).count(8), '\n' "9:", tuple(labels[:, 1]).count(9), '\n' "10:",
    #       tuple(labels[:, 1]).count(10))
    # print("0:", tuple(labels[:, 0]).count(0), '\n' "1:", tuple(labels[:, 0]).count(1), '\n' "2:",
    #       tuple(labels[:, 0]).count(2), '\n' "3:", tuple(labels[:, 0]).count(3))

    # 数据集处理
    if data_name == 'test':
        dataset = Data.TensorDataset(features, labels)
    elif data_name == 'origin':
        dataset = Data.TensorDataset(features, labels[:, 0])
    else:
        dataset = Data.TensorDataset(features, labels[:, 1])
    return dataset, num_sample


# data_process('test')


def dataset_process(dataset, num_sample):
    num_test = int(rate_test * num_sample)
    num_train = num_sample - num_test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test], generator=None)

    # for tset
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
    return train_iter, test_iter


# def _set_init(layer):
#     init.normal_(layer.weight, mean=0., std=.1)
#     init.constant_(layer.bias, B_INIT)


def vgg_block(num_convs, in_channels, out_channels):  # vgg卷积层
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.BatchNorm2d(in_channels))
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        # blk.append(nn.Dropout(p=0.1))
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


def fc_block(num_fcs, input_size, num_classes):  # 全连接层
    fcs = []
    hidden_size = input_size // 4
    for i in range(num_fcs):
        if i == 0 and num_fcs != 1:
            fcs.append(nn.Linear(input_size, hidden_size))
            fcs.append(nn.BatchNorm1d(hidden_size))
            fcs.append(nn.ReLU())
            # fcs.append(nn.Dropout(p=0.1))
        elif i == 0:
            fcs.append(nn.Linear(input_size, num_classes))
            fcs.append(nn.Softmax(dim=1))
        elif i != (num_fcs - 1):
            fcs.append(nn.Linear(hidden_size, hidden_size))
            fcs.append(nn.BatchNorm1d(hidden_size))
            fcs.append(nn.ReLU())
            # fcs.append(nn.Dropout(p=0.1))
        else:
            fcs.append(nn.Linear(hidden_size, num_classes))
            fcs.append(nn.Softmax(dim=1))
    return nn.Sequential(*fcs)


class CNN(nn.Module):
    def __init__(self, num_convs,
                 in_channels,
                 out_channels,
                 num_fcs,
                 input_size,
                 num_classes,
                 ):
        super(CNN, self).__init__()
        self.cnn = vgg_block(num_convs, in_channels, out_channels)
        in_size = out_channels * (input_size[0] // 2 ** num_convs) * (input_size[1] // 2 ** num_convs)
        self.out = fc_block(num_fcs, in_size, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, out_channels * input_size[0] *
        # input_size[1])//(2**num_convs))
        output = self.out(x)
        return output


def train_and_save(str, train_iter, test_iter):
    cnn_origin = CNN(num_convs=1, in_channels=1, out_channels=17, num_fcs=2, input_size=(18, 17), num_classes=4)
    cnn_coverage = CNN(num_convs=1, in_channels=1, out_channels=11, num_fcs=2, input_size=(18, 17), num_classes=5)
    cnn_capacity = CNN(num_convs=1, in_channels=1, out_channels=6, num_fcs=2, input_size=(18, 17), num_classes=4)
    cnn_interference = CNN(num_convs=1, in_channels=1, out_channels=5, num_fcs=1, input_size=(18, 17), num_classes=4)
    if str == "origin":
        cnn = cnn_origin
    elif str == "coverage":
        cnn = cnn_coverage
    elif str == "capacity":
        cnn = cnn_capacity
    else:
        cnn = cnn_interference

    # print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)  # optimize all cnn parameters
    train_ch5(cnn, train_iter, test_iter, optimizer, device, num_epochs)
    # torch.save(cnn.state_dict(), 'cnn_' + str + '.pkl')  # save only the parameters
    torch.save(cnn, 'cnn_' + str + '.pt')  # save the model


def evaluate(str, x_sample, y_sample):
    cnn = torch.load('cnn_' + str + '.pt')
    cnn.to(device)
    cnn.eval()
    # cnn.load_state_dict(torch.load('cnn_' + str + '.pkl'))
    x_sample = x_sample.view(1, 1, 18, 17)
    y_hat2_sample = cnn(x_sample)
    y_hat2_max_sample = y_hat2_sample.argmax(dim=1)
    # if str == 'interference':
    #     print(y_hat2_max_sample.item(), (y_sample - 7).item())
    if str == 'capacity' and y_sample != 0:
        acc_num = (y_hat2_max_sample == y_sample - 4).float().to(device).item()
    elif str == 'interference' and y_sample != 0:
        acc_num = (y_hat2_max_sample == y_sample - 7).float().to(device).item()
    else:
        acc_num = (y_hat2_max_sample == y_sample + 0).float().to(device).item()
    return acc_num


def load_and_predict():
    cnn = torch.load('cnn_origin.pt')
    cnn = cnn.to(device)
    cnn.eval()
    # cnn.load_state_dict(torch.load('cnn_origin.pkl'))
    dataset, _ = data_process('test')
    data_iter = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
        pin_memory=True,
        drop_last=False
    )
    origin_acc_sum, coverage_acc_sum, capacity_acc_sum, interference_acc_sum, n, n1, n2, n3 = 0.0, 0.0, 0.0, 0.0, \
        0, 0, 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.float().to(device)
            y = y.to(device)
            y_hat1 = cnn(X)
            y_hat1_max = y_hat1.argmax(dim=1)
            origin_acc_sum += (y_hat1_max == y[:, 0] + 0).float().sum().to(device).item()
            for index, (x_sample, y_hat1_max_sample) in enumerate(zip(X, y_hat1_max)):
                if y_hat1_max_sample == 1:
                    coverage_acc_sum += evaluate('coverage', x_sample, y[index, 1])
                    n1 += 1
                elif y_hat1_max_sample == 2:
                    capacity_acc_sum += evaluate('capacity', x_sample, y[index, 1])
                    n2 += 1
                elif y_hat1_max_sample == 3:
                    interference_acc_sum += evaluate('interference', x_sample, y[index, 1])
                    n3 += 1
            n += y.shape[0]
            print('origin acc %.3f, coverage acc %.3f, capacity acc %.3f, interference acc %.3f'
                  % (origin_acc_sum / n, coverage_acc_sum / n1, capacity_acc_sum / n2, interference_acc_sum / n3))


def train_model(str):
    dataset, num_sample = data_process(str)
    train_iter, test_iter = dataset_process(dataset, num_sample)
    train_and_save(str, train_iter, test_iter)


# train_model('capacity')
load_and_predict()
