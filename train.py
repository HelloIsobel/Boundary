from Final.new_unet_model import U_net
from Final.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tool.value2excel import getExcel
import time
import cv2

# 定义训练函数
def evaluate_loss_acc(data_iter, net, device):
    # 计算data_iter上的平均损失与准确率
    loss = nn.BCEWithLogitsLoss()  # TODO: nn.CrossEntropyLoss()
    is_training = net.training  # Bool net是否处于train模式
    net.eval()
    l_sum, acc_sum, n = 0, 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = net(X)
            l = loss(y_hat, y)
            l_sum += l.item() * y.shape[0]
            # acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
    net.train(is_training)  # 恢复net的train/eval状态
    # return l_sum / n, acc_sum / n
    print("You are pig!!!")
    return l_sum / n


### 评价模型准确率
def evaluate_accuracy(data_iter, net, criterion, device):
    valid_l_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = net(X)
            loss = criterion(y_hat, y)
            valid_l_sum += loss.cuda().item()
            n += y.shape[0]
    print("You are pig!!!")
    return valid_l_sum / n  # 总准确率/样本


def train_net(net, device, data_path, val_path, epochs=100, batch_size=1, lr=0.00001):  # 原来40 1 0.00001
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    isbi_dataset_val = ISBI_Loader(val_path)
    val_loader = torch.utils.data.DataLoader(dataset=isbi_dataset_val,
                                             batch_size=batch_size,
                                             shuffle=True)

    net.to(device=device)
    # print("training on ", device)

    # # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)  # 原来
    criterion = nn.BCEWithLogitsLoss()  # TODO: nn.CrossEntropyLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    train_l_value = []
    valid_l_value = []

    # 训练epochs次
    for epoch in range(epochs):
        train_l_sum, n, train_l, start = 0.0, 0.0, 0.0, time.time()
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        print('epoch:', epoch)
        for image, label in train_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            label_hat = net(image)
            loss = criterion(label_hat, label)
            print('Loss/train:', loss.item())

            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), '0218_bestmodel.pth')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.cuda().item()
            n += label.shape[0]
        train_l = train_l_sum / n
        test_l = evaluate_accuracy(val_loader, net, criterion, device)
        print('epoch %d, loss %.4f, test loss %.4f, time %.1f sec'
              % (epoch + 1, train_l, test_l, time.time() - start))
        train_l_value.append(train_l)
        valid_l_value.append(test_l)

    excel_path = '/home/zxy/3-Boundary_detection/Unet/data/0218excel_loss_train.xlsx'
    getExcel(train_l_value, excel_path, header=False)
    excel_path1 = '/home/zxy/3-Boundary_detection/Unet/data/0218excel_loss_valid.xlsx'
    getExcel(valid_l_value, excel_path1, header=False)

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = U_net(1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path1 = "/home/zxy/3-Boundary_detection/Unet/data/train/"
    val_path1 = '/home/zxy/3-Boundary_detection/Unet/data/val/'
    train_net(net, device, data_path1, val_path1)
