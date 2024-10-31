import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image


class Dataloader_SUSE(Dataset):
    def __init__(self, root, transforms, names=['satellite', 'drone']):
        super(Dataloader_SUSE).__init__()
        self.transforms_drone_street = transforms['train']   # 加载数据增强操作
        self.transforms_satellite = transforms['satellite']
        self.root = root
        self.names = names
        hights = ['150', '200', '250', '300']
        dict_path = {}      # 存放 satellite图片绝对路径的list + drone图片绝对路径的list
        for hight in hights:
            for name in names:  # 依次遍历 names=['satellite', 'drone']
                dict_ = {}
                for cls_name in os.listdir(os.path.join(root, hight, name)):          # cls_name 表示的是图片的类别 一个类别文件夹里面存放着drone或sate图像
                    img_list = os.listdir(os.path.join(root, hight, name, cls_name))  # 图片的名字 1006.jpg
                    img_path_list = [os.path.join(root, hight, name, cls_name, img) for img in img_list]  # 一个类别的所有图像绝对地址
                    dict_[cls_name] = img_path_list
                dict_path[name] = dict_

        # 获取设置名字与索引之间的镜像
        cls_names = os.listdir(os.path.join(root, hights[0], names[0]))         # 根据 satellite 中的类别获取类别名
        cls_names.sort()                                             # 将类别名由小到大排序
        map_dict = {i: cls_names[i] for i in range(len(cls_names))}  # {0:'0839', 1:'0842', ...}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        self.index_cls_nums = 2

    # 从指定的视角+类别中随机抽一张出来 返回PIL读取的图像和类别（1354）
    def sample_from_cls(self, name, cls_num):
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path, 1)[0]
        # 在data_1中因为存在image-02-2.jpeg 所以这样截取ID会出现问题 但是因为 label_id 没用到所以没关系
        label_id = img_path.split('/')[-1].split('.')[0].split('-')[-1]
        img = Image.open(img_path).convert("RGB")
        return img, label_id

    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        img, _ = self.sample_from_cls("satellite", cls_nums)
        img_s = self.transforms_satellite(img)
        img, label_id_dr = self.sample_from_cls("drone", cls_nums)
        img_d = self.transforms_drone_street(img)
        return img_s, img_d, index, label_id_dr  # 卫星图 无人机图 类别索引（也就是训练标签）第四项没有用

    def __len__(self):
        return len(self.cls_names)  # 返回类别len


class Sampler_University(object):  # 选取一个batch的图片 并设置每个样本采样的个数

    def __init__(self, data_source, batch_size=8, sample_num=4):
        self.data_len = len(data_source)
        self.batchsize = batch_size
        self.sample_num = sample_num
        self.data_source = data_source

    def __iter__(self):
        data_list = np.arange(0, self.data_len)
        np.random.shuffle(data_list)
        nums = np.repeat(data_list, self.sample_num, axis=0)
        return iter(nums)

    def __len__(self):
        return len(self.data_source)


# collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
def train_collate_fn(batch):
    img_s, img_d, ids, label_id_dr = zip(*batch)
    ids = torch.tensor(ids, dtype=torch.int64)
    return [torch.stack(img_s, dim=0), ids, label_id_dr], [torch.stack(img_d, dim=0), ids, label_id_dr]


if __name__ == '__main__':
    transform_train_list = [
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_train_list = {"satellite": transforms.Compose(transform_train_list), "train": transforms.Compose(transform_train_list)}
    datasets = Dataloader_SUSE(root="../data/SUES-200-512x512/Training", transforms=transform_train_list, names=['satellite', 'drone'])
    samper = Sampler_University(datasets, 8)
    dataloader = DataLoader(datasets, batch_size=8, num_workers=0, sampler=samper, collate_fn=train_collate_fn)
    for data_s, data_d in dataloader:
        inputs2, labels2, label_num = data_d
        print("labels2", labels2)
        print("label_num", label_num)


