from tools.random_erasing import RandomErasing
from tools.autoaugment import ImageNetPolicy
from torchvision import datasets, transforms
import os
from triplet_samp_load.Dataloader_University import Dataloader_University, Sampler_University, train_collate_fn
from triplet_samp_load.Dataloader_SUSE import Dataloader_SUSE
from triplet_samp_load.Dataloader_CVUSA import Dataloader_CVUSA
import torch


def get_data_loader(h, w, pad, erasing_p, color_jitter, DA, data_dir, sample_num, batchsize, data_name, add_weather):
    transform_train_list = [
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(pad, padding_mode='edge'),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    street_transform_train_list = [  # 112 616
        transforms.Resize((112, 616), interpolation=3),
        transforms.Pad(pad, padding_mode='edge'),
        transforms.RandomCrop((112, 616)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_satellite_list = [
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(pad, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # 是否添加随机擦除的数据增强方法
    if erasing_p > 0:
        transform_train_list = transform_train_list + [RandomErasing(probability=erasing_p, mean=[0.0, 0.0, 0.0])]

    # 是否调用 color_jitter 数据颜色增强方法
    if color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                       hue=0)] + transform_train_list
        transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                           hue=0)] + transform_satellite_list

    # 是否使用基于NAS的数据增强方法 auto augmentation 包含对于颜色的增强
    if DA:
        transform_train_list = [ImageNetPolicy()] + transform_train_list

    # 生成数据增强的字典
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'satellite': transforms.Compose(transform_satellite_list),
        'street': transforms.Compose(street_transform_train_list)
    }

    # 生成数据地址 添加数据增强 并生成字典 drone 换成 street
    if data_name == 'SUSE':
        image_datasets = {
            'satellite': datasets.ImageFolder(os.path.join(data_dir, '150', 'satellite'), data_transforms['satellite']),
        }
    elif data_name == 'University_1652':
        image_datasets = {
            'satellite': datasets.ImageFolder(os.path.join(data_dir, 'satellite'), data_transforms['satellite']),
        }
    elif data_name == 'CVUSA':
        image_datasets = {
            'satellite': datasets.ImageFolder(os.path.join(data_dir, 'satellite'), data_transforms['satellite']),
        }

    # 获取数据集各个部分的大小并打印
    dataset_sizes = len(image_datasets['satellite']) * sample_num

    # choose dataset
    if data_name == 'University_1652':
        image_datasets_tri = Dataloader_University(data_dir, transforms=data_transforms, add_weather=add_weather)

    elif data_name == 'SUSE':
        image_datasets_tri = Dataloader_SUSE(data_dir, transforms=data_transforms)

    elif data_name == 'CVUSA':
        image_datasets_tri = Dataloader_CVUSA(data_dir, transforms=data_transforms)


    # 三元组采样
    samper = Sampler_University(data_source=image_datasets_tri, batch_size=batchsize, sample_num=sample_num)
    dataloaders = torch.utils.data.DataLoader(image_datasets_tri, batch_size=batchsize, sampler=samper,
                                              num_workers=8,
                                              pin_memory=True, collate_fn=train_collate_fn)

    num_classes = len(image_datasets['satellite'].classes)
    return dataloaders, num_classes, dataset_sizes

