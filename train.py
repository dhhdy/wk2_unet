import torch
import os
import time
import datetime
from Unet import UNet
from my_dataset import Data_Loader
from torch.utils.data import Dataset
from torch import optim
from torch import nn
def train_net(net, device, train_path, pred_path, epochs=150, batch_size=2, lr=0.00001):
    train_dataset = Data_Loader(train_path)
    pred_dataset = Data_Loader(pred_path)
    # 测试数据读取是否正常
    # print(train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=pred_dataset,
                                               batch_size=1,
                                               shuffle=True)
    # 定义优化器
    optimiser = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    length = len(train_dataset)
    # 定义损失函数
    criterion = nn.BCEWithLogitsLoss()

    # best_loss统计，初始化为正无穷
    best_acc = float('inf')
    # 训练epochs次
    for epoch in range(epochs):

        running_loss = 0.0
        # 训练模式
        net.train()

        # 按照batch_size训练
        for img, lbl in train_loader:
            # 梯度清零
            optimiser.zero_grad()

            # 数据拷贝到device上运行
            img = img.to(device=device, dtype=torch.float32)
            lbl = lbl.to(device=device, dtype=torch.float32)

            # 使用网络参数，输出预测结果
            pred = net(img)
            # 计算loss, loss是评估指标
            loss = criterion(pred, lbl)
            # loss的数值, loss.item()获取对应py类型，只是数值运算下使用，能不消耗内存节省开销
            # print('Loss_train', loss.item())
            # 计算损失和
            running_loss += loss.item()
            # 更新参数
            loss.backward()
            optimiser.step()

        net.eval()
        acc = 0.0  # 正确率
        total = 0
        with torch.no_grad():
            for img, lbl in test_loader:
                img = img.to(device=device, dtype=torch.float32)
                lbl = lbl.to(device=device, dtype=torch.float32)
                test_pred = net(img)
                test_pred[test_pred > 0] = 1
                test_pred[test_pred <= 0] = 0
                # print(test_pred)
                # print(lbl)
                # print((test_pred == lbl).sum().item())
                # print(lbl.size(2) * lbl.size(3))
                acc += (test_pred == lbl).sum().item() / (lbl.size(2) * lbl.size(3))
                total += lbl.size(0)
                accurate = acc / total
                # 保存最小loss对应的参数
                if accurate > best_acc:
                    best_acc = accurate
                    torch.save(net.state_dict(), 'best_model.pth')

        print('epoch:', epoch + 1, 'loss_avg: ', running_loss / length, 'accurate: ', accurate * 100, '%')

if __name__ == '__main__':

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络，图片通道3，分类为1。
    net = UNet(n_channels=3, n_classes=1)

    # 将网络拷贝到device中
    net.to(device=device)

    # 指定训练集地址，开始训练
    # print(os.path.abspath(data_path))
    train_net(net, device, './DIRVE/training', './DIRVE/prediction')