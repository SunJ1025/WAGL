import torch
import torch.nn.functional as F
import torch.nn as nn


# 代替attention层的无参数傅立叶变换模块Fourier
class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


class GeMPool(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    we add flatten and norm so that we can use it as one aggregation layer.
    """

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        # ------------------------- 添加 频域 ---------------------- #
        # self.fft = FNetBlock()
        # ------------------------- 添加 频域 ---------------------- #

    def forward(self, x):
        # ------------------------- 添加 频域 ---------------------- #
        # x = self.fft(x)+x
        # ------------------------- 添加 频域 ---------------------- #
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)
        x = x.flatten(1)
        return F.normalize(x, p=2, dim=1)


if __name__ == '__main__':
    gem = GeMPool()
    b = torch.randn(8, 2048, 8, 8)
    out_put = gem(b)
    print(out_put.shape)  # torch.Size([8, 2048]) size no change
