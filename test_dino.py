# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from convap import ConvAP

import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import time
import scipy.io
import yaml
import os
from tools.utils import which_view, get_id, extract_feature_dino, load_dino_network
from typing import Literal
from dino_extract import DinoV2ExtractFeatures
# from dinov2 import DINOv2

# test dino
from Dino_models.helper import VPRModel

parser = argparse.ArgumentParser(description='Training')

# 获取测试集地址
parser.add_argument('--test_dir', default='./data/University-1652/test', type=str, help='test data path')
# 输出模型的名字
parser.add_argument('--name', default='trained_model_name', type=str, help='save model path')
# 测试使用的 batchsize 大小
parser.add_argument('--batchsize', default=128, type=int, help='batch size ')
# 图像高 默认为 256
parser.add_argument('--h', default=252, type=int, help='height')
# 图像宽 默认为 256
parser.add_argument('--w', default=252, type=int, help='width')
# 选择测试方式
parser.add_argument('--mode', default='1', type=int, help='1: satellite->drone  2: drone->satellite')
# 是否使用re-rank
parser.add_argument('--re_rank', default=1, type=int, help='1表示使用 0表示不使用')
# 是否chose epoch to test
parser.add_argument('--epoch', default='last', help='chose epoch to test or the final epoch')

opt = parser.parse_args()
re_rank = opt.re_rank
test_dir = opt.test_dir

# 设置 GPU
torch.cuda.set_device(0)
cudnn.benchmark = True
use_gpu = torch.cuda.is_available()

h_new, w_new = (opt.h // 14) * 14, (opt.w // 14) * 14
# 数据预处理 resize 和归一化
data_transforms = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.CenterCrop((h_new, w_new)),
])

# 加载测试数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(test_dir, x), data_transforms)
                  for x in ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone', ]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=4)
               for x in ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']}

print('-------test-----------')
print(opt.name)

since = time.time()  # 开始计时

# 根据需要选取查询集和待查集
if opt.mode == 1:
    query_name = 'query_satellite'
    gallery_name = 'gallery_drone'
elif opt.mode == 2:
    query_name = 'query_drone'
    gallery_name = 'gallery_satellite'
else:
    raise Exception("opt.mode is not required")

# 获取对应的编号
which_gallery = which_view(gallery_name)
which_query = which_view(query_name)
print('查询集： %s -> 待查集： %s:' % (query_name.split('_')[1], gallery_name.split('_')[1]))

# 写入 gallery name
save_path = f'evaluation/weights/{opt.name}'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

gallery_path = image_datasets[gallery_name].imgs
query_path = image_datasets[query_name].imgs

# 获取 gallery 和 query 的 类别标签以及图像路径
gallery_label, gallery_path = get_id(gallery_path)
query_label, query_path = get_id(query_path)


if __name__ == "__main__":
    # 提取特征
    with torch.no_grad():

        extractor = VPRModel(
            # ---- Encoder
            backbone_arch='dinov2_vitl14',  # dinov2_vitl14 resnet101 res-ibn-50
            backbone_config={
                # 'num_trainable_blocks': 4,
                # 'return_token': True,  # False
                # 'norm_layer': True,
            },
            agg_arch='Gem',  # SALAD AFIM Gem
            agg_config={
                # 'num_channels': 1024,
                # 'num_clusters': 64,
                # 'cluster_dim': 128,
                # 'token_dim': 256,
                # 'channel': 1024,    # AFIM=1024
            },
            # class_num=opt.nclasses
        ).cuda().eval()

        # load trained pth
        extractor = load_dino_network(opt, extractor)
        print("get extractor")

        query_feature = extract_feature_dino(extractor, dataloaders[query_name], which_query)
        gallery_feature = extract_feature_dino(extractor, dataloaders[gallery_name], which_gallery)

    time_elapsed = time.time() - since
    print('Test feature extract complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    result = {'name': opt.name,
              'query_name': query_name,
              'gallery_name': gallery_name,
              'gallery_f': gallery_feature.numpy(),
              'gallery_label': gallery_label,
              'gallery_path': gallery_path,
              'query_f': query_feature.numpy(),
              'query_label': query_label,
              'query_path': query_path
              }

    scipy.io.savemat(os.path.join(save_path, f'{query_name}_result_{opt.epoch}.mat'), result)
    print("save_mat:", f'{query_name}_result_{opt.epoch}.mat')

    # 将结果保存在 model 文件夹里面的 txt 文件里
    result = 'evaluation/weights/%s/result.txt' % opt.name
    # 调用 evaluate_gpu.py 文件进行评估
    os.system(f'python evaluation/evaluate_cpu_no_rerank.py --epoch {opt.epoch} --model_name {opt.name} --view {opt.mode}| tee -a %s' % result)
