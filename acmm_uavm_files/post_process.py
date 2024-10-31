import pandas as pd
import scipy.io
import torch
import numpy as np
import time

import sys
import os.path as osp
import scipy.io
import ast

sys.path.append(osp.join(osp.dirname(__file__), ".."))

# ------------------------------------------------- 读取提供的 query_drone_name.txt ---------------------------#
name_rank = []
with open("./query_drone_name.txt", "r") as f:
    for txt in f.readlines():
        name_rank.append(txt[:-1])
print(name_rank[0:10])

# 从txt文件读取列表
query_img_list = []
gallery_img_list = []
with open("query_img_list.txt", "r") as file:
    for line in file:
        query_img_list.append(ast.literal_eval(line.strip()))

with open("gallery_img_list.txt", "r") as file:
    for line in file:
        gallery_img_list.append(ast.literal_eval(line.strip()))

print(query_img_list[0:10])
print(gallery_img_list[0:10])

# ----------------------------------------- post process  -------------------------------------#
csv_name = 'agg_3'
# save_path_1 = f'/home/oem/桌面/drone/TriSSA/mat_files/acmmm_no_graph_gem_bs_16_eph_60_dino_l_252_add_weather_drone_result_last.mat'
# save_path_2 = f'/home/oem/桌面/drone/TriSSA/mat_files/acmmm_gem_dino-l-448-add_weather_drone_result_last.mat'
save_path_1 = f'/home/oem/桌面/drone/TriSSA/mat_files/acmmm_gem_dinol_392_di-add_weather_drone_result_last.mat'
save_path_2 = f'/home/oem/桌面/drone/TriSSA/mat_files/acmmm_gem_convx-b-384-add_weather-200-0.3lr_drone_result_last.mat'

result = scipy.io.loadmat(save_path_1)
query_feature1, gallery_feature1 = torch.FloatTensor(result['query_f']), torch.FloatTensor(result['gallery_f'])

result2 = scipy.io.loadmat(save_path_2)
query_feature2, gallery_feature2 = torch.FloatTensor(result2['query_f']), torch.FloatTensor(result2['gallery_f'])


print("loaded feature")

# 定义模型权重
w1, w2 = 0.8, 0.8  # 0.8, 0.7, 0.7, 0.6

result = {}

result = {}
from collections import defaultdict
vote_result = defaultdict(int)
from tqdm import tqdm

for i in tqdm(range(len(query_img_list))):
    # for i in range(len(query_img_list)):
    # 获取每个模型的查询特征
    query1 = query_feature1[i].view(-1, 1)
    query2 = query_feature2[i].view(-1, 1)

    # 计算每个模型的相似度分数
    score1 = torch.mm(gallery_feature1, query1).squeeze(1).cpu().numpy()
    score2 = torch.mm(gallery_feature2, query2).squeeze(1).cpu().numpy()

    # 获取每个模型的排序索引
    # index1 = np.argsort(score1)[::-1]
    # index2 = np.argsort(score2)[::-1]
    # index3 = np.argsort(score4)[::-1]
    #
    # # 对每个模型的前10个结果进行投票
    # for idx in index1[:10]:
    #     vote_result[idx] += 1
    # for idx in index2[:10]:
    #     vote_result[idx] += 1
    # for idx in index3[:10]:
    #     vote_result[idx] += 1
    #
    # # 根据投票结果排序
    # final_index = sorted(vote_result.keys(), key=lambda x: vote_result[x], reverse=True)
    # max_score_list = final_index[:10]

    # 加权平均相似度分数
    combined_score = w1 * score1 + w2 * score2
    # 获取排序索引
    index = np.argsort(combined_score)[::-1]
    # 获取前10个最相关的图像索引
    max_score_list = index[:10]

    # 获取查询图像路径
    query_img = query_img_list[i][0]

    # 初始化最相关的图像列表
    most_correlative_img = []

    for idx in max_score_list:
        most_correlative_img.append(gallery_img_list[idx][0])

    # 将查询图像及其最相关的图像存储到结果字典中
    result[query_img] = most_correlative_img

    # 清空投票结果
    vote_result.clear()

matching_table = pd.DataFrame(result)
matching_table.to_csv(f"result_{csv_name}_rerank.csv")
print("save csv finish")

# ----------------------------------------- export  -------------------------------------#
with open("./query_drone_name.txt", "r") as f:
    txt = f.readlines()
    f.close()
txt = [i.split("\n")[0] for i in txt]

table = pd.read_csv(f"result_{csv_name}_rerank.csv", index_col=0)
result = {}
for i in table:
    result[i.split("/")[-1]] = [k.split("/")[-1].split(".")[0] for k in list(table[i])]

with open(f"answer_{csv_name}_rerank.txt", "w") as p:
    for t in txt:
        p.write(' '.join(result[t]))
        p.write("\n")
