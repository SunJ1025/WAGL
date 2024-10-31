import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib.pyplot as plt

# demo.py 是直接 torch.mm 计算的相似度分数 不具备参考意义

# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=55, type=int, help='test_image_index')      # index 是从0000文件夹开始排序的
parser.add_argument('--test_dir', default='../data/test', type=str, help='./test_data')
opts = parser.parse_args()


gallery_name = 'gallery_satellite'     # 确定 query 和 gallery 需要注意根据mat文件确定query和gallery
query_name = 'query_drone'
# gallery_name = 'gallery_drone'
# query_name = 'query_satellite'

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in [gallery_name, query_name]}


def imshow(path, title=None):    # show 图像
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated


result = scipy.io.loadmat('../pytorch_result.mat')          # 加载保存的结果数据
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]
query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

multi = os.path.isfile('multi_query.mat')                  # 是否存在多重查询的结果
if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()


# sort the images
def sort_img(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    score_index = np.argsort(score)[::-1]  # predict index from small to large  # index = index[0:2000]

    junk_index = np.argwhere(gl == -1)     # 排除错误数据
    mask = np.in1d(score_index, junk_index, invert=True)
    score_index = score_index[mask]

    return score_index


i = opts.query_index
index = sort_img(query_feature[i], query_label[i], gallery_feature, gallery_label)

# Visualize the rank result
query_path, _ = image_datasets[query_name].imgs[i]
query_label = query_label[i]
print(query_path)
print('Top 10 images are as follow:')

save_folder = './%02d' % opts.query_index       # 保存 query 图像
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
os.system('cp %s %s/query.jpg' % (query_path, save_folder))

# Visualize Ranking Result
try:
    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 11, 1)
    ax.axis('off')
    imshow(query_path, 'query')
    for i in range(10):
        ax = plt.subplot(1, 11, i+2)
        ax.axis('off')
        img_path, _ = image_datasets[gallery_name].imgs[index[i]]
        label = gallery_label[index[i]]
        print("gallery label:", label)
        imshow(img_path)
        os.system('cp %s %s/s%02d.jpg' % (img_path, save_folder, i))
        if label == query_label:
            ax.set_title('%d' % (i+1), color='green')
        else:
            ax.set_title('%d' % (i+1), color='red')
        print(img_path)
    plt.pause(100)
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig(f"{save_folder}/show.png")

