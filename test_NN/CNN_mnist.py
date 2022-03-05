from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models.Nets import *
import argparse

if __name__ == '__main__':
    writer = SummaryWriter('../logs')

    trans_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据
    dataset_train = datasets.MNIST("../data", train=True, transform=trans_mnist, download=True)
    dataset_test = datasets.MNIST("../data", train=False, transform=trans_mnist, download=True)
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    test_data_size = len(dataset_test)

    # args()需要的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    args = parser.parse_args()
    # 构建网络
    net_glob = CNNMnist(args=args)
    print(net_glob)

    # 优化器和损失函数
    optimizer = optim.SGD(net_glob.parameters(), lr=0.01, momentum=0.5)
    loss_fn = nn.CrossEntropyLoss()

    # 训练的轮数
    epoch = 20
    total_train_step = 0
    total_test_step = 0

    # training
    list_loss = []
    net_glob.train()
    for i in range(epoch):
        print("----------第{}次训练----------".format(i + 1))
        for data in train_loader:
            imgs, targets = data
            output = net_glob(imgs)
            loss = loss_fn(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("训练次数:{}，loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar("CNNmnsit_train_loss", loss.item(), total_train_step)

        # testing
        net_glob.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                output = net_glob(imgs)
                loss = loss_fn(output, targets)
                total_test_loss = total_test_loss + loss
                accuracy = (output.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

            print("整体测试集上的loss:{}".format(total_test_loss))
            print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
            writer.add_scalar("CNNmnsit_test_loss", total_test_loss, total_test_step)
            writer.add_scalar("CNNmnsit_test_accuracy", total_accuracy / test_data_size, total_test_step)
            total_test_step = total_test_step + 1

writer.close()
