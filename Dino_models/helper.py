import numpy as np
from Dino_models import aggregators
from Dino_models import backbones
import torch.nn as nn
from torch.autograd import Variable
import torch
from modules.ClassBlock import ClassBlock
from torch.nn import functional as F
from modules.attention import GraphAttentionLayer

from aggregators.Adaptive_Feature_Integration_Module import AFIM

# return backbone models
def get_backbone(
        backbone_arch='dinov2',
        backbone_config={}
):
    if 'resnet50' in backbone_arch.lower():  # or 'resnet101'
        print(backbone_arch.lower())
        print("use resnet")
        return backbones.ResNet(backbone_arch, **backbone_config)

    elif 'res-ibn-50' in backbone_arch.lower():
        print("use res-ibn-50")
        return backbones.resnet50_ibn_a()

    elif 'res-ibn-101' in backbone_arch.lower():
        print("use res-ibn-101")
        return backbones.resnet101_ibn_a()

    elif 'dinov2' in backbone_arch.lower():
        print("use dino")
        return backbones.DINOv2(model_name=backbone_arch, **backbone_config)

    elif 'convnext' in backbone_arch.lower():
        print("use dino")
        return backbones.convnext_base()


# return aggregate layer
def get_aggregator(
        agg_arch='ConvAP',
        agg_config={}
):
    if 'cosplace' in agg_arch.lower():
        assert 'in_dim' in agg_config
        assert 'out_dim' in agg_config
        return aggregators.CosPlace(**agg_config)

    elif 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)

    elif 'convap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.ConvAP(**agg_config)

    elif 'mixvpr' in agg_arch.lower():
        assert 'in_channels' in agg_config
        assert 'out_channels' in agg_config
        assert 'in_h' in agg_config
        assert 'in_w' in agg_config
        assert 'mix_depth' in agg_config
        return aggregators.MixVPR(**agg_config)

    elif 'salad' in agg_arch.lower():
        assert 'num_channels' in agg_config
        assert 'num_clusters' in agg_config
        assert 'cluster_dim' in agg_config
        assert 'token_dim' in agg_config
        return aggregators.SALAD(**agg_config)

    elif 'afim' in agg_arch.lower():
        assert 'channel' in agg_config
        return aggregators.AFIM(**agg_config)

    elif 'netvlad' in agg_arch.lower():
        assert 'num_clusters' in agg_config
        assert 'dim' in agg_config
        return aggregators.NetVLAD(**agg_config)


class VPRModel(nn.Module):
    def __init__(self,
                 # ---- Backbone
                 backbone_arch='resnet50',
                 use_graph=True,
                 backbone_config={},

                 # ---- Aggregator
                 agg_arch='ConvAP',
                 agg_config={},

                 # Linear BN reLU Linear
                 class_num=701,
                 droprate=0.5,
                 circle=True
                 ):
        super(VPRModel, self).__init__()
        # Backbone
        self.encoder_arch = backbone_arch
        self.use_graph=use_graph
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # get the backbone and the aggregator
        self.backbone = get_backbone(backbone_arch, backbone_config)
        self.aggregator = get_aggregator(agg_arch, agg_config)

        # classifier  AFIM+l=2048 salad+l=8448 res50+gem=2048 gem+l=1024 res50-ibn+gem=2048 dino-g=1536 convnext=1024
        self.classifier = ClassBlock(1024, class_num, droprate, return_f=circle)

        # --------------------- 添加新的分类头 ----------------------------#
        # self.classifier_local = ClassBlock(1024, class_num, droprate, num_bottleneck=512, return_f=circle)
        # self.local_branch = AFIM(channel=1024)
        # --------------------- 添加新的分类头 ----------------------------#

    def forward(self, x1, x2):

        if x1 is None:
            y1 = None
        else:
            x1 = self.backbone(x1)  # gem+resnet[16, 1024, 8, 8]
            # ---------------- 局部分支 ---------------------------- #
            # x1_loc = self.local_branch(x1)
            # y1_loc = self.classifier_local(x1_loc)
            # ---------------- 局部分支 ---------------------------- #
            x1 = self.aggregator(x1)  # [16, 1024]
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.backbone(x2)  # [16, 1024, 8, 8]
            # ---------------- 局部分支 ---------------------------- #
            # x2_loc = self.local_branch(x2)
            # y2_loc = self.classifier_local(x2_loc)
            # ---------------- 局部分支 ---------------------------- #
            x2 = self.aggregator(x2)
            y2 = self.classifier(x2)  # y1 y2 都包含两部分： 1. 通过第二层FC之后的概率值  2. 中间维度的特征值

        if self.training:

            # half_size = y1[1].size(0)
            # first_half = x_g_4_fuse[:half_size]
            # second_hale = x_g_4_fuse[half_size:]
            # x_g_4_fuse = first_half + second_hale
            # x_g_4_fuse = self.low_dim(x_g_4_fuse.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
            #
            # logits, ff = y1
            # logits2, ff2 = y2
            # ff = torch.cat((ff, x_g_4_fuse), dim=1)  # y1[1] torch.Size([8, 512]) torch.Size([16, 2048])
            # ff2 = torch.cat((ff2, x_g_4_fuse), dim=1)
            # y1 = logits, ff
            # y2 = logits2, ff2

            return y1, y2
        else:
            return x1, x2
            # if x1 is None:
            #    return x1, torch.cat((x2, x2_loc), dim=1)
            # if x2 is None:
            #    return torch.cat((x1, x1_loc), dim=1), x2


if __name__ == '__main__':
    model = VPRModel(
        # ---- Encoder
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': False,
            'norm_layer': True,
        },
        agg_arch='AFIM',      # NetVLAD
        agg_config={
            # 'num_clusters': 120,
            # 'dim': 1024,
            'channel': 768,  # one branch dim
        },
        class_num=100
    ).eval()

    input_img = Variable(torch.FloatTensor(8, 3, 252, 252))
    adj = torch.randn(16, 16)
    output = model(input_img, input_img, adj)
    print(output[1].shape)
    # print('net output size:', output.shape)
