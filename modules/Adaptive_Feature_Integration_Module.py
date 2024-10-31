import torch
import torch.nn as nn


class AFIM(nn.Module):
    def __init__(self, channel=2048, reduction=16):
        super().__init__()

        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # nn.BatchNorm1d(channel// reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction , channel , bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, h, w = x.size()

        x1 = self.avgpool2(x)
        x1 = x1.view([b, c])
        x1 = self.fc(x1)
        x1 = x1.view([b, c, 1, 1])
        x1_out = x*x1
        x1_out = self.avgpool(x1_out) 

        x2 = self.maxpool2(x)
        x2 = x2.view([b, c])
        x2 = self.fc(x2)
        x2 = x2.view([b, c, 1, 1])
        x2_out = x*x2
        x2_out = self.maxpool(x2_out) 

        out_all = torch.cat((x1_out, x2_out), dim=1)
        out_all = out_all.view(out_all.size(0), out_all.size(1))
        return out_all 


if __name__ == '__main__':
    afim = AFIM()
    b = torch.randn(8, 2048, 8, 8)
    out_put = afim(b)
    print(out_put.shape)  # torch.Size([8, 4096])
