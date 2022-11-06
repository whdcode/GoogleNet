import torch
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


# 单独定义二维卷积层的类
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# 定义inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 最大池化不改变输入通道数
            BasicConv2d(in_channels, pool_proj, kernel_size=1)  # 保证输出大小等于输入大小
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        outputs = [x1, x2, x3, x4]
        return torch.cat(outputs, dim=1)  # NCHW中的第一维


# 定义辅助分类器

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # out；[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        """当实例化一个模型之后，使用model.train() 和model.eval()来控制模型状态，
        model.train()中，self.training = Ture表示在训练中生效，model.eval()中,
        self.training = False表示在测试中生效"""
        x = F.dropout(x, p=0.5, training=self.training)  # 原文p = 0.7
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N X 1024
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        # N x classes_nums
        return x


class GoogleNet(nn.Module):
    def __init__(self, nums_classes=1000, aux_logits=True, init_weights=False):  # aux_logits=True是否使用辅助分类器
        super(GoogleNet, self).__init__()
        self.aux_logits = aux_logits  # 定义一个判断是否需要辅助分类器的标号

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 这里的输出特征图尺寸进行了下取整out = N X 64 x 112 x112
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.localres1 = nn.LocalResponseNorm(size=5)  # 局部响应归一化

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.localres2 = nn.LocalResponseNorm(size=5)

        self.maxpool2 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1_result = InceptionAux(512, nums_classes)  # from 4a out 512 x 4 x 4
            self.aux2_result = InceptionAux(528, nums_classes)  # from 4d out 528 x 4 x 4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均下采样, 不考虑输入尺寸，都输出目标（1,1）的输出尺寸
        # self.avgpool = nn.Averagepool(7, 1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, nums_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # input_size = [N x 3 x 224 x 224]
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.localres1(x)
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.localres2(x)
        x = self.maxpool2(x)
        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:  # 处于训练状态且需要辅助分类器时有这层
            aux1 = self.aux1_result(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2_result(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x classes_num
        if self.training and self.aux_logits:
            return x, aux2, aux1
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    writer = SummaryWriter('test_net_graph')
    net = GoogleNet()
    input = torch.ones((1, 3, 224, 224))
    writer.add_graph(net, input)
    print(net)
    writer.close()
