
import scipy.io
import torch
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

nums = 54*10 # 如果 query 是 drone 的话 每个类别有54张图 15表示选取15个类别

file_path = '../evaluation/weights/drone_2_sallite.mat'
result = scipy.io.loadmat(file_path)

query_feature = torch.FloatTensor(result['query_f'])      # [37855, 4096]
query_label = result['query_label'][0]                    # [0,0,0,... 1,1,1,...] 54 one group
gallery_feature = torch.FloatTensor(result['gallery_f'])  # [951, 4096]
gallery_label = result['gallery_label'][0]                # [  0    1    2    3    4    5 ...]


tsne = TSNE(n_components=3)                               # n_components 2或者3
test_tsne = tsne.fit_transform(query_feature[:nums])

fig = plt.figure(figsize=(8, 8))            # 画图
ax = fig.add_subplot(projection='3d')		# 创建3D子图

# r = 0.6
# pi = np.pi
# cos = np.cos
# sin = np.sin
# phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
# x = r*sin(phi)*cos(theta)
# y = r*sin(phi)*sin(theta)
# z = r*cos(phi)
# ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])

ax.set_ylim(ymin=-0.6, ymax=0.6)            # 固定X轴、Y轴的范围
ax.set_xlim(xmin=-0.6, xmax=0.6)

x_min, x_max = np.min(test_tsne, 0), np.max(test_tsne, 0)    # 归一化
test_tsne = test_tsne / (x_max - x_min)                      # (2160, 3) (nums, 3)


# softmax = torch.nn.Softmax(dim=1)
# test_tsne = softmax(torch.from_numpy(test_tsne)).numpy()

from sklearn.cluster import KMeans
# 使用K均值聚类找到类中心
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(test_tsne)
centers = kmeans.cluster_centers_
# 计算每个点到最近的类中心的距离
distances = np.linalg.norm(test_tsne - centers[kmeans.labels_], axis=1)
# print(distances.shape)
# print(distances)
print(query_label[:nums].shape)
pos_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
n, bins, _ = plt.hist(query_label[:nums], bins=539, alpha=0.7, density=True, color=pos_color, label='distances')
plt.plot(bins, distances, color=pos_color)  # plot y curve


## ax.scatter(test_tsne[:, 0], test_tsne[:, 1], c=query_label[0:nums:2], cmap=plt.cm.Spectral)              # 2d图 根据标签上颜色
# ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', marker='X', s=20, label='Cluster Centers')
# ax.scatter(test_tsne[:, 0], test_tsne[:, 1], test_tsne[:, 2], c=query_label[:nums], cmap=plt.cm.Spectral, s=20)  # 3d图
# ax.view_init(5, -72)		# 初始化视角
# # plt.axis('off')             # 关闭坐标系
#
# sa_name = file_path.split('/')[-1].split('.mat')[0]          # 保存图像
# fig.savefig(f'{sa_name}_3d.png', dpi=300)
plt.show()
