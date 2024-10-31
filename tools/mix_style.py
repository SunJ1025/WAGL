import random
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6, samp_num=4):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.samp_num = samp_num
    
    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})'

    def forward(self, x, x2):
        if not self.training:
            return x
        if random.random() > self.p:
            return x
        
        # 虽然输入的是x x2 但是 x2 只是被用来 mix 返回的是 mix 之后的 x1
        # 先把两部分数据拼接起来
        x = torch.cat((x, x2), dim=0)
        B = x.size(0)

        # x 归一化
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        # mu2 = x2.mean(dim=[2, 3], keepdim=True)
        # var2 = x2.var(dim=[2, 3], keepdim=True)
        # sig2 = (var2 + self.eps).sqrt()
        # mu2, sig2 = mu2.detach(), sig2.detach()
        # x2_normed = (x - mu2) / sig2
        
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.arange(B)          # 随机打乱 这里是两个batch
        perm_a, perm_b = perm.chunk(2)  # 分成两个部分
   
        perm_a = [x_p[torch.randperm(self.samp_num)] for x_p in perm_a.chunk(int(B/(2*self.samp_num)))]
        perm_a = torch.cat(perm_a, 0)

        perm_b = [x_p[torch.randperm(self.samp_num)] for x_p in perm_b.chunk(int(B/(2*self.samp_num)))]
        perm_b = torch.cat(perm_b, 0)

        perm = torch.cat([perm_a, perm_b], dim=0)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        x_out = x_normed * sig_mix + mu_mix
        x_out_1, x_out_2 = torch.split(x_out, int(B/2), dim=0)

        return x_out_1


if __name__ == '__main__':
    x = torch.randn([16, 3, 16, 16], dtype=torch.float32)
    x2 = torch.randn([16, 3, 16, 16], dtype=torch.float32)

    mix = MixStyle()
    out = mix(x, x2)
    print(out.shape)
