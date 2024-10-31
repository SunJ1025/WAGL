import torch
import os
from torchvision import datasets, transforms
import argparse
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import time
from PIL import Image
import cv2
import scipy.io
import torch
import numpy as np
import time
from evaluation.re_ranking import re_ranking

from Dino_models.helper import VPRModel
from tools.utils import which_view, extract_feature_dino, extract_feature_dino_defog, extract_feature_dino_no_defog, load_dino_network
from tools.utils import getRecoverScene

import sys
import os.path as osp
import scipy.io

sys.path.append(osp.join(osp.dirname(__file__), ".."))

parser = argparse.ArgumentParser(description='Training')

# ------------------------------------------------- 测试时需要修改 ---------------------------------------------- #
# 1. --name
# 2. --h --w
# 3. --defog
# 4. csv_name
# 5. --re_rank

csv_name = 'gem_dino-l-448-add_weather'  # gem_convx-b-384-add_weather

# ------------------------------------------------- 修改参数的位置 ---------------------------------------------- #
# 输出模型的名字
parser.add_argument('--name', default='gem_dino-l-448-add_weather', type=str,
                    help='save model path')
# 测试使用的 batchsize 大小
parser.add_argument('--batchsize', default=64, type=int, help='batch size ')
# 图像高 默认为 256
parser.add_argument('--h', default=448, type=int, help='height')
# 图像宽 默认为 256
parser.add_argument('--w', default=448, type=int, help='width')
# 选择测试方式
parser.add_argument('--mode', default='2', type=int, help='1: satellite->drone  2: drone->satellite')
# 是否使用re-rank
parser.add_argument('--re_rank', default=1, type=int, help='1表示使用 0表示不使用')
# 是否chose epoch to test
parser.add_argument('--epoch', default='last', help='chose epoch to test or the final epoch')
# 是否在测试时去雾
parser.add_argument('--defog', default=0, type=int, help='the first K epoch that needs warm up')
# ------------------------------------------------- 修改参数的位置 ---------------------------------------------- #

opt = parser.parse_args()
re_rank = opt.re_rank

# 设置 GPU
torch.cuda.set_device(0)
cudnn.benchmark = True
use_gpu = torch.cuda.is_available()

# ------------------------------------------------- 读取提供的 query_drone_name.txt ---------------------------#
name_rank = []
with open("../acmmm2024/query_drone_name.txt", "r") as f:
    for txt in f.readlines():
        name_rank.append(txt[:-1])
print(name_rank[0:10])


# ------------------------------------------------- 读取无人机数据的 dataset ---------------------------#
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, file_list, transform=None, target_transform=None, de_fog=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self._make_dataset(file_list)
        self.defog = de_fog
        # print(self.samples)

    def _make_dataset(self, file_list):
        data = []
        for line in file_list:
            path = os.path.join(self.root, "query_drone160k_wx", "query_drone_160k_wx_24", line)
            item = (path, int(0))
            data.append(item)
        return data

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.defog:
            # --------------------- add defog -------------------------- #
            sample_np = np.array(sample)[..., ::-1]  # RGB -> BGR
            sample_np = cv2.resize(sample_np, (opt.h, opt.w))
            transmission, sample_de = getRecoverScene(sample_np)
            # cv2.imwrite('/home/oem/桌面/drone/TriSSA/' + 'DCP.jpg', sample_de)
            sample_de = Image.fromarray(sample_de[..., ::-1])
            sample_de.save('sample_de.jpg')
            if self.transform is not None:
                sample_de = self.transform(sample_de)
            # --------------------- add defog -------------------------- #

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.defog:
            sample_de = sample

        return sample, sample_de, target


transform_test_list = [
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),  # 会默认归一化到0-1
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {'drone': transforms.Compose(transform_test_list),
                   'satellite': transforms.Compose(transform_test_list)}
image_datasets = {
    'satellite': datasets.ImageFolder(os.path.join("/home/oem/桌面/drone/OneDrive_1_2024-5-11", 'gallery'),
                                      data_transforms['satellite'])}

# ------------------------------------------------- 判断是否去雾 -------------------------------------#
if opt.defog > 0:
    defog_ = True
    print("use defog")
else:
    defog_ = False
    print("no use of defog")
image_datasets['drone'] = CustomImageFolder(os.path.join("/home/oem/桌面/drone/OneDrive_1_2024-5-11", 'query'),
                                            name_rank,
                                            data_transforms['drone'],
                                            de_fog=defog_)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=8, pin_memory=True)
               for x in ['satellite', 'drone']}

with open('../acmmm2024/query_drone_name.txt', 'r') as f:
    order = [line.strip() for line in f.readlines()]
image_datasets['drone'].imgs = sorted(image_datasets['drone'].imgs, key=lambda x: order.index(x[0].split("/")[-1]))

# ------------------------------------ 提取特征 ------------------------------------- #
since = time.time()  # 开始计时
# 根据需要选取查询集和待查集
if opt.mode == 1:
    query_name = 'satellite'
    gallery_name = 'drone'
elif opt.mode == 2:
    query_name = 'drone'
    gallery_name = 'satellite'
else:
    raise Exception("opt.mode is not required")

# 获取对应的编号
which_gallery = which_view(gallery_name)
which_query = which_view(query_name)
print('查询集： %s -> 待查集： %s:' % (query_name, gallery_name))

with torch.no_grad():
    extractor = VPRModel(
        # ---- Encoder
        backbone_arch='dinov2_vitl14',  # dinov2_vitl14 dino模型需要  backbone_config 且 'return_token': False
        # backbone_config={
        #     'num_trainable_blocks': 4,
        #     'return_token': False,  # False
        #     'norm_layer': True,
        # },
        agg_arch='Gem',  # SALAD AFIM Gem
        agg_config={
            # 'num_channels': 1024,
            # 'num_clusters': 64,
            # 'cluster_dim': 128,
            # 'token_dim': 256,
            # 'channel': 1024,    #  AFIM+l=1024
        },
        # class_num=opt.nclasses
    ).cuda().eval()

    # load trained pth

    extractor = load_dino_network(opt, extractor)
    print("get extractor")
    if opt.defog > 0:
        query_feature = extract_feature_dino_defog(extractor, dataloaders[query_name], which_query)
    else:
        query_feature = extract_feature_dino_no_defog(extractor, dataloaders[query_name], which_query)
    gallery_feature = extract_feature_dino(extractor, dataloaders[gallery_name], which_gallery)

time_elapsed = time.time() - since
print('Test feature extract complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# ----------------------------------------- generate results  -------------------------------------#
query_img_list = image_datasets["drone"].imgs
gallery_img_list = image_datasets["satellite"].imgs

# 写入列表到txt文件
# with open("query_img_list.txt", "w") as file:
#     for item in query_img_list:
#         file.write(f"{item}\n")
#
# with open("gallery_img_list.txt", "w") as file:
#     for item in gallery_img_list:
#         file.write(f"{item}\n")


result = {}
for i in range(len(query_img_list)):
    query = query_feature[i].view(-1, 1)      # torch.Size([1024, 1])
    score = torch.mm(gallery_feature, query)  # torch.Size([168437, 1])
    score = score.squeeze(1).cpu()
    index = np.argsort(score.numpy())
    index = index[::-1].tolist()
    max_score_list = index[0:10]
    query_img = query_img_list[i][0]
    most_correlative_img = []
    for index in max_score_list:
        most_correlative_img.append(gallery_img_list[index][0])
    result[query_img] = most_correlative_img

matching_table = pd.DataFrame(result)

matching_table.to_csv(f"result_{csv_name}.csv")
print("save csv finish")

# ---------------------------- save mat and evaluate --------------------------------- #
result = {'name': opt.name,
          'query_name': query_name,
          'gallery_name': gallery_name,
          'gallery_f': gallery_feature.numpy(),
          'query_f': query_feature.numpy(),
          }

save_path = f'evaluation/weights/{opt.name}/acmmm_{csv_name}_{query_name}_result_{opt.epoch}.mat'
scipy.io.savemat(save_path, result)
print("save_mat:", f'acmmm_{csv_name}_{query_name}_result_{opt.epoch}.mat')

# ----------------------------------------- export  -------------------------------------#
with open("../acmmm2024/query_drone_name.txt", "r") as f:
    txt = f.readlines()
    f.close()
txt = [i.split("\n")[0] for i in txt]

table = pd.read_csv(f"result_{csv_name}.csv", index_col=0)
result = {}
for i in table:
    result[i.split("/")[-1]] = [k.split("/")[-1].split(".")[0] for k in list(table[i])]

with open(f"answer_{csv_name}.txt", "w") as p:
    for t in txt:
        p.write(' '.join(result[t]))
        p.write("\n")
print("export finish")

