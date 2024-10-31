import torch
import torch.nn as nn
from torch.autograd import Variable
from Dino_models.backbones.resnet_ibn import resnet50_ibn_a
from modules.Adaptive_Feature_Integration_Module import AFIM
from modules.ClassBlock import ClassBlock


class net_ibn(nn.Module):
    def __init__(self, stride=2):
        super(net_ibn, self).__init__()
        model_ft_ibn = resnet50_ibn_a(last_stride=stride, pretrained=False)
        model_ft_ibn.load_param('pre_weights/resnet50_ibn_a.pth.tar')
        self.afim = AFIM()
        self.model = model_ft_ibn

    def forward(self, x):

        x = self.model(x)  # ([16, 2048, 8, 8])
        x = self.afim(x)
        x = x.view(x.size(0), x.size(1))  # torch.Size([16, 2048])

        return x


# 将backbone整合 组成整体的网络 并选择是否权重共享
class two_view_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, share_weight=True, circle=True):
        super(two_view_net, self).__init__()

        self.model_1 = net_ibn(stride=stride)

        # 如果是共享权重的话 就只初始化一次
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = net_ibn(stride=stride)

        self.circle = circle
        self.classifier = ClassBlock(4096, class_num, droprate, return_f=circle)

    def forward(self, x1, x2):
        # 根据是否输入数据选择不同分支的模型 不过如果是共享权重的话 model_1 model_2 都是一样的 判断也就没有必要了
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)     # x1 x2 为模型的输出 还没经过全连接层
            y2 = self.classifier(x2)  # y1 y2 都包含两部分： 1. 通过第二层FC之后的概率值  2. 中间维度的特征值
        if self.training:
            return y1, y2             # x1, x2  # 原始代码没有判断 self.training 直接 return y1, y2
        else:
            return x1, x2


if __name__ == '__main__':
    net = two_view_net(701, droprate=0.5).eval()
    input = Variable(torch.FloatTensor(16, 3, 256, 256))
    output1, output2 = net(input, None)
    print('net output size:', output1.shape)
