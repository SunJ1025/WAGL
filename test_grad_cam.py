
import argparse
import numpy as np
import cv2

import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image

from tools.utils import load_dino_network
from Dino_models.helper import VPRModel


parser = argparse.ArgumentParser(description='Training')
# 获取测试集地址
parser.add_argument('--test_dir', default='./data/University-1652/test', type=str, help='test data path')
# 输出模型的名字
parser.add_argument('--name', default='gem_resibn50_252', type=str)
# gem_resibn101_252 train_university_no_graph_gem_bs_16_eph_120_dino_l_252  train_university_no_graph_gem_bs_16_eph_60_dino_l_252_add_weather
# 测试使用的 batchsize 大小
parser.add_argument('--batchsize', default=128, type=int, help='batch size ')
# 选择测试方式
parser.add_argument('--mode', default='1', type=int, help='1: satellite->drone  2: drone->satellite')
# 是否使用re-rank
parser.add_argument('--re_rank', default=1, type=int, help='1表示使用 0表示不使用')
# 是否chose epoch to test
parser.add_argument('--epoch', default='last', help='chose epoch to test or the final epoch')

opt = parser.parse_args()


extractor = VPRModel(
    # ---- Encoder
    backbone_arch='res-ibn-50',  # dinov2_vitl14 resnet101 res-ibn-50 res-ibn-101
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
)


def print_model_structure(model, indent=0):
    # 获取模块的所有子模块（层）
    for name, module in model.named_children():
        # 打印缩进和模块名称
        print(' ' * indent + f'{name}: {module.__class__.__name__}')
        # 递归打印子模块的层
        print_model_structure(module, indent + 2)

# load trained pth
model = load_dino_network(opt, extractor)
model.eval()
print_model_structure(model)
traget_layers = [model.backbone]


# 1.定义模型结构，选取要可视化的层
# resnet18 = models.resnet18(pretrained=True)
# resnet18.eval()
# traget_layers = [resnet18.layer4]

# 2.读取图片，将图片转为RGB
origin_img = cv2.imread('/home/oem/桌面/drone/OneDrive_1_2024-5-11/query/query_drone160k_wx/query_drone_160k_wx_24/0AyTW60A18vxzu7.jpeg')
# origin_img = cv2.imread('/home/oem/桌面/drone/OneDrive_1_2024-5-11/gallery/gallery_satellite_160k/0AD9uIptRbTXM6h.webp')
# origin_img = cv2.imread('/home/oem/桌面/drone/TriSSA/data/University-1652/test/query_drone/0001/image-09.jpeg')

# 0A150rlFPWGtArj.jpeg 0A8yphNEn0jx093.jpeg 0BrZnmkat4oEhCr.jpeg
rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

# 3.图片预处理：resize、裁剪、归一化
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(392),
    transforms.CenterCrop(392)
])
crop_img = trans(rgb_img)
net_input = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop_img).unsqueeze(0)

# 4.将裁剪后的Tensor格式的图像转为numpy格式，便于可视化
canvas_img = (crop_img*255).byte().numpy().transpose(1, 2, 0)
canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)

# 5.实例化cam
cam = pytorch_grad_cam.GradCAMPlusPlus(model=model, target_layers=traget_layers)
grayscale_cam = cam(net_input)
grayscale_cam = grayscale_cam[0, :]

cam.batch_size = 3

# 6.将feature map与原图叠加并可视化
src_img = np.float32(canvas_img) / 255
visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
cv2.imshow('feature map', visualization_img)
cv2.waitKey(0)