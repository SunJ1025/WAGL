
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
ax = fig.add_subplot()		# 创建3D子图


ax.set_ylim(ymin=0, ymax=0.2)            # 固定X轴、Y轴的范围
ax.set_xlim(xmin=0, xmax=15)

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

# max value of each class from the center
y = np.array([])
cls_nums = 54
for i in range(n_clusters):
    distace_cls = np.linalg.norm(test_tsne[i*cls_nums:(i+1)*cls_nums,] - centers[kmeans.labels_[i*cls_nums:(i+1)*cls_nums,]], axis=1)
    distance_max = np.max(distace_cls)
    y = np.append(y, distance_max)

pos_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
n, bins, _ = plt.hist(query_label[:nums], bins=539, alpha=0.7, density=False, color=pos_color, label='distances')
# plt.plot(bins, distances, color=pos_color)  # plot y curve
print(bins)
print(distances)
plt.show()
