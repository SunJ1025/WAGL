import torch.nn as nn
import torch

# 代替attention层的无参数傅立叶变换模块Fourier
class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x

# FNet结构图中的Feed Forward模块 实则就是一个mlp
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# LayerNorm模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.norm = nn.InstanceNorm2d(dim)

        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# FNet网络
class FNet(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # print(attn, ff)
            x = attn(x) + x
            print(x.shape)
            print(ff)
            print(ff(x).shape, x.shape)
            x = ff(x) + x
        return x
    
x = torch.randn(4,3,256,256)
FNet = FNet(dim=3,depth=3,mlp_dim=128)
out = FNet(x)
print(out.shape)