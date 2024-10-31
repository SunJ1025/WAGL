import torch
import torch.nn as nn


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


class AFIM(nn.Module):
    def __init__(self, channel=768, reduction=16):
        super().__init__()

        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # nn.BatchNorm1d(channel// reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.IBN = IBN(planes=1024)

    def forward(self, x):
        b, c, h, w = x.size()  # [16, 1024, 18, 18]

        x = self.IBN(x)

        x1 = self.avgpool2(x)
        x1 = x1.view([b, c])
        x1 = self.fc(x1)
        x1 = x1.view([b, c, 1, 1])
        x1_out = x * x1
        x1_out = self.avgpool(x1_out)

        x2 = self.maxpool2(x)
        x2 = x2.view([b, c])
        x2 = self.fc(x2)
        x2 = x2.view([b, c, 1, 1])
        x2_out = x * x2
        x2_out = self.maxpool(x2_out)

        # out_all = torch.cat((x1_out, x2_out), dim=1)
        out_all = x1_out
        out_all = out_all.view(out_all.size(0), out_all.size(1))
        return out_all


if __name__ == '__main__':
    afim = AFIM()
    b = torch.randn(2, 768, 18, 18)
    out_put = afim(b)
    print(out_put.shape)  # torch.Size([2, 1536])
