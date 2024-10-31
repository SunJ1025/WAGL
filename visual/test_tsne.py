
import scipy.io
import torch

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 每次 test 就会生成 mat 文件
# mat 文件根据每次 test 的 query 和 gallery 会保存所有的 feature

nums = 54*15                          # 如果 query 是 drone 的话 每个类别有54张图 15表示选取15个类别

file_path = '../matfiles/no_tri.mat'  # mat文件路径 res-ibn mean_0.9 no_tri no_cd_tri
result = scipy.io.loadmat(file_path)  # 加载mat文件 query_name gallery_name gallery_f gallery_label gallery_path query_f query_label query_path

query_feature = torch.FloatTensor(result['query_f'])     # 加载 query 和 gallery 的 feature 以及 label
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

tsne = TSNE(n_components=2)                               # n_components 2或者3
test_tsne = tsne.fit_transform(query_feature[0:nums:2])   # 隔一个特征选取一个 避免可视化杂乱 也就是每个类别选取27张

fig = plt.figure(figsize=(8, 8))            # 画图
ax = fig.add_subplot()                      # projection='2d' 默认为2d

ax.set_ylim(ymin=-0.6, ymax=0.6)  # 固定X轴、Y轴的范围
ax.set_xlim(xmin=-0.6, xmax=0.6)

x_min, x_max = np.min(test_tsne, 0), np.max(test_tsne, 0)    # 归一化
test_tsne = test_tsne / (x_max - x_min)

ax.scatter(test_tsne[:, 0], test_tsne[:, 1], c=query_label[0:nums:2], cmap=plt.cm.Spectral)              # 2d图 根据标签上颜色

sa_name = file_path.split('/')[-1].split('.mat')[0]          # 保存图像
fig.savefig(f'{sa_name}_2d.png', dpi=300)
plt.show()
