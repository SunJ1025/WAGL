import torch
import torch.nn as nn

import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), ".."))


DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

# class IBN(nn.Module):
#     r"""Instance-Batch Normalization layer from
#     `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
#     <https://arxiv.org/pdf/1807.09441.pdf>`
#
#     Args:
#         planes (int): Number of channels for the input tensor
#         ratio (float): Ratio of instance normalization in the IBN layer
#     """
#
#     def __init__(self, planes, ratio=0.5):
#         super(IBN, self).__init__()
#         self.half = int(planes * ratio)
#         self.IN = nn.InstanceNorm2d(self.half, affine=True)
#         self.BN = nn.BatchNorm2d(planes - self.half)
#
#     def forward(self, x):
#         split = torch.split(x, self.half, 1)
#         out1 = self.IN(split[0].contiguous())
#         out2 = self.BN(split[1].contiguous())
#         out = torch.cat((out1, out2), 1)
#         return out

class DINOv2(nn.Module):

    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False  # If True, the forward pass returns both the feature map and the token
    ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        # self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = torch.hub.load('/home/oem/桌面/drone/TriSSA/facebookresearch_dinov2_main', model_name,
                                    source='local')
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

        # -------------- add IBN ----------------- #
        # self.IBN = IBN(planes=1024)

    def forward(self, x):    # [B, 3, H, W] -> [B, C, H // 14, W // 14]  token [B, C].

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        # f = self.IBN(f) + f

        if self.return_token:
            return f, t
        return f



