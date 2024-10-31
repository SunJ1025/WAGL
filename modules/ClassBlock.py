import torch.nn as nn
from torch.nn import init
import torch


# 凯明初始化
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# 对分类层初始化
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# 当inplace=True时： 就是将从上层网络 nn.Conv2d 中传递下来的 tensor 直接进行修改，这样能够节省运算内存，不用多存储其他变量
def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


# in_dim --> 512 BN Drop 512-->701
class ClassBlock(nn.Module):  # 全连接层 BN层 relu 全连接层
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=True):
        super(ClassBlock, self).__init__()
        self.return_f = return_f

        add_block = []
        # 添加线性层
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim

        # 添加BN层
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
            # add_block += [IBN(num_bottleneck)]

        # 添加 ReLU
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]

        # 将以上层组织到 Sequential 中并进行初始化
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        # 初始化分类层
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        # 判断是否需要返回feature
        if self.return_f:
            # 预留没有经过最后分类层的 feature
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x
