# CIFAR-10 图像分类实验：基于 PyTorch 的 CNN 模型
import torch
import torchvision.datasets
import time
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备训练集
train_data=torchvision.datasets.CIFAR10("../data/dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
# 准备测试数据集
test_data=torchvision.datasets.CIFAR10("../data/dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)
# 创建网络模型，记得from model import *
class Beverly(nn.Module):
    def __init__(self):
        super(Beverly, self).__init__()
        self.moudl1 = Sequential(nn.Conv2d(3,32,5,1,2),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(32,32,5,1,2),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(32,64,5,1,2),
                                 nn.MaxPool2d(2),
                                 nn.Flatten(),
                                 nn.Linear(64*4*4,64),
                                 nn.Linear(64,10)
        )

    def forward(self,x):
        x=self.moudl1(x)
        return x
beverly = Beverly()
if torch.cuda.is_available():
    beverly=beverly.cuda()
#损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()
# 优化器
leaning_rate=1e-2
optimizer=torch.optim.SGD(beverly.parameters(),lr=leaning_rate)
# 设置训练网络的参数
total_train_step=0#记录训练次数
total_test_step=0#记录测试次数
epoch=10#记录训练轮数
#添加tensorboard
writer=SummaryWriter("../logs/logs_modeltrain")
start_time=time.time()
for i in range(10):
    print("-----第{}轮训练开始-------".format(i+1))
    beverly.train()
    for data in train_dataloader:
        imgs,targets=data
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()
        output1=beverly(imgs)
        loss1=loss_fn(output1,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        total_train_step=total_train_step+1
        if total_train_step%100==0:
            end_time=time.time()
            print(end_time-start_time)
            print("训练次数：{}，Loss：{}".format(total_train_step,loss1))
            writer.add_scalar("train_loss",loss1.item(),total_train_step)
    #测试步骤开始
    beverly.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            if torch.cuda.is_available():
                targets = targets.cuda()
                imgs = imgs.cuda()
            output2=beverly(imgs)
            accuracy=(output2.argmax(1)==targets).sum()
            loss2=loss_fn(output2,targets)
            total_test_loss=loss2.item()+total_test_loss
            total_accuracy=total_accuracy+accuracy
            total_test_step = total_test_step + 1
    print("测试次数：{}，整体测试集的损失：{},整体测试集正确率：{}".format(total_test_step,total_test_loss,total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    torch.save(beverly,"Beverly_{}.pth".format(i))
    print("模型已保存")
writer.close()
