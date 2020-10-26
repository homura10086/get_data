import torch
import numpy as np
import time
from matplotlib import pyplot as plt
import torch.nn.functional as F

# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#         n += y.shape[0]
#     return acc_sum / n


def evaluate_accuracy_2(data_iter, net, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X.float().to(device)).argmax(dim=1) == y.to(device)).float().sum().to(device).item()
            net.train()  # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


def train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    train_acc, test_acc, train_loss = [], [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.float().to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.to(device).item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().to(device).item()
            n += y.shape[0]
            batch_count += 1
        test_acc.append(evaluate_accuracy_2(test_iter, net, device))
        train_acc.append(train_acc_sum / n)
        train_loss.append(train_l_sum / batch_count)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss[epoch], train_acc[epoch], test_acc[epoch], time.time() - start))
    # 绘图
    plt.plot(range(num_epochs), test_acc, linewidth=2, color='olivedrab', label='test data')
    plt.plot(range(num_epochs), train_acc, linewidth=2, color='chocolate', linestyle='--', label='train data')
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    test_max_index = np.argmax(test_acc).item()
    show_max = '[' + str(test_max_index) + ', ' + str(round(test_acc[test_max_index], 3)) + ']'
    # 以●绘制最大值点和最小值点的位置
    plt.plot(test_max_index, test_acc[test_max_index], 'ko')
    plt.annotate(show_max, xy=(test_max_index, test_acc[test_max_index]),
                 xytext=(test_max_index, test_acc[test_max_index]))
    plt.grid()
    plt.show()
    plt.plot(range(num_epochs), train_loss, linewidth=2, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max - min)


def index_generate(num, batch_size, device):
    return torch.ones(batch_size, dtype=torch.int64).view(-1, 1).to(device) * num


def softmax(data_iter, device, net, batch_size):
    net = net.to(device)
    data_acc1, data_acc2, data_acc3, data_acc4, n = 0, 0, 0, 0, 0
    data_acc11, data_acc21, data_acc31, data_acc41, n1 = 0, 0, 0, 0, 0
    data_acc12, data_acc22, data_acc32, data_acc42, n2 = 0, 0, 0, 0, 0
    data_acc13, data_acc23, data_acc33, data_acc43, n3 = 0, 0, 0, 0, 0
    for X, y in data_iter:
        X = X.float().to(device)
        y = y.to(device)
        y_hat = net(X)
        y_hat_softmax = F.softmax(y_hat, dim=1)  # 各样本概率分布
        y_hat_order = y_hat_softmax.argsort(dim=1)  # 各样本概率分布从小到大的索引排列
        y_hat_order_max1 = torch.gather(y_hat_order, dim=1, index=index_generate(3, batch_size, device))
        y_hat_order_max2 = torch.gather(y_hat_order, dim=1, index=index_generate(2, batch_size, device))
        y_hat_order_max3 = torch.gather(y_hat_order, dim=1, index=index_generate(1, batch_size, device))
        for i in range(batch_size):
            if y[i] == 0:
                n += 1
                if y_hat_order_max1[i] == 0:
                    data_acc1 += 1
                elif y_hat_order_max2[i] == 0:
                    data_acc2 += 1
                elif y_hat_order_max3[i] == 0:
                    data_acc3 += 1
                else:
                    data_acc4 += 1
            elif y[i] == 1:
                n1 += 1
                if y_hat_order_max1[i] == 1:
                    data_acc11 += 1
                elif y_hat_order_max2[i] == 1:
                    data_acc21 += 1
                elif y_hat_order_max3[i] == 1:
                    data_acc31 += 1
                else:
                    data_acc41 += 1
            elif y[i] == 2:
                n2 += 1
                if y_hat_order_max1[i] == 2:
                    data_acc12 += 1
                elif y_hat_order_max2[i] == 2:
                    data_acc22 += 1
                elif y_hat_order_max3[i] == 2:
                    data_acc32 += 1
                else:
                    data_acc42 += 1
            else:
                n3 += 1
                if y_hat_order_max1[i] == 3:
                    data_acc13 += 1
                elif y_hat_order_max2[i] == 3:
                    data_acc23 += 1
                elif y_hat_order_max3[i] == 3:
                    data_acc33 += 1
                else:
                    data_acc43 += 1
    print('y = 0')
    print(data_acc1 / n)
    print(data_acc2 / n)
    print(data_acc3 / n)
    print(data_acc4 / n)
    print('y = 1')
    print(data_acc11 / n1)
    print(data_acc21 / n1)
    print(data_acc31 / n1)
    print(data_acc41 / n1)
    print('y = 2')
    print(data_acc12 / n2)
    print(data_acc22 / n2)
    print(data_acc32 / n2)
    print(data_acc42 / n2)
    print('y = 3')
    print(data_acc13 / n3)
    print(data_acc23 / n3)
    print(data_acc33 / n3)
    print(data_acc43 / n3)