import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleCNN(nn.Module):
   def __init__(self):
       super(SimpleCNN, self).__init__()
       # 卷积层1，输入通道1，输出通道10，卷积核大小5
       self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
       # 卷积层2，输入通道10，输出通道20，卷积核大小5
       self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
       # 池化层，使用2x2最大池化
       self.pooling = nn.MaxPool2d(2)
       # 全连接层，输入特征320，输出特征10
       self.fc = nn.Linear(320, 10)
   def forward(self, x):
       # 应用卷积层1，激活函数ReLU，然后池化
       x = F.relu(self.pooling(self.conv1(x)))
       # 应用卷积层2，激活函数ReLU，然后池化
       x = F.relu(self.pooling(self.conv2(x)))
       # 将特征图展平
       x = x.view(-1, 320)
       # 应用全连接层
       x = self.fc(x)
       return x
# 创建模型实例
model = SimpleCNN()