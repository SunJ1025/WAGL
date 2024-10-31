# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from tqdm import tqdm
import time
import os

# from model import two_view_net
import yaml
from shutil import copyfile
from tools.utils import save_network, setup_seed
from losses.triplet_loss import Tripletloss
from losses.cal_loss import cal_triplet_loss
from losses.cross_entroy_loss import cross_entropy_loss
from pytorch_metric_learning import losses, miners
from triplet_samp_load.Get_DataLoader import get_data_loader
import torchvision
import math

# test dino
from Dino_models.helper import VPRModel

version = torch.__version__
parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--name', default='test_final', type=str, help='output model name')
parser.add_argument('--data_dir', default='./data/University-1652/train', type=str, help='training dir path')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--num_epochs', default=120, type=int, help='')
parser.add_argument('--data_name', default='University_1652', type=str, help='')

parser.add_argument('--pad', default=0, type=int, help='padding')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')

parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--sample_num', default=4, type=float, help='num of repeat sampling')
parser.add_argument('--margin', default=0.3, type=float, help='num of margin')

parser.add_argument('--labelsmooth', default=0, type=int, help='1表示使用 0表示不使用')
parser.add_argument('--share', action='store_true', help='share weight between different view')

parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')  # 设置 FC层的 drop rate
parser.add_argument('--stride', default=1, type=int, help='stride')  # 网络最后层下采样的步长 默认为2

parser.add_argument('--warm_epoch', default=5, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--steps', default=[30, 50], type=int, help='')  # 学习率突变的 epochs [70, 110]
parser.add_argument('--balance', default=1.0, type=float, help='balance rate for triplet loss')  # 平衡三元组损失正则化项

# action='store_true' 如果添加 --triplet 就表示使用
parser.add_argument('--triplet', action='store_true', help='use triplet loss')
parser.add_argument('--arcface', action='store_true', help='use ArcFace loss')
parser.add_argument('--circle', action='store_true', help='use Circle loss')
parser.add_argument('--cosface', action='store_true', help='use CosFace loss')
parser.add_argument('--contrast', action='store_true', help='use contrast loss')
parser.add_argument('--lifted', action='store_true', help='use lifted loss')
parser.add_argument('--sphere', action='store_true', help='use sphere loss')

parser.add_argument('--use_graph', action='store_true', help='whether use graph or not')
parser.add_argument('--add_weather', default=1, type=int, help='if add_w = 1 add')

opt = parser.parse_args()

# 设置 GPU
torch.cuda.set_device(0)
cudnn.benchmark = True
setup_seed()

# 判断是否添加 weather
if opt.add_weather == 1:
    add_weather = True
else:
    add_weather = False
dataloaders, opt.nclasses, dataset_sizes = get_data_loader(opt.h, opt.w, opt.pad, opt.erasing_p, opt.color_jitter,
                                                           opt.DA, opt.data_dir, opt.sample_num, opt.batchsize, opt.data_name, add_weather=add_weather)
print("类别数：", opt.nclasses)


total_iters = opt.num_epochs * 176


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, use_graph=True):  # 训练模型

    begin = time.time()

    warm_up = 0.1  # warm_up 训练策略设置
    warm_iteration = round(dataset_sizes / opt.batchsize) * opt.warm_epoch  # first 5 epoch
    MAX_LOSS = 10

    if opt.triplet:
        triplet_loss = Tripletloss(margin=opt.margin, balance=opt.balance)  # 域间三元组
        miner = miners.MultiSimilarityMiner()  # 域内三元组
        criterion_triplet = losses.TripletMarginLoss(margin=0.3)

    for epoch in tqdm(range(num_epochs), desc="Processing"):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        since = time.time()
        model.train(True)

        running_cls_loss = 0.0  # 每一个 epoch 将 loss 清零
        running_triplet = 0.0
        running_graph = 0.0
        running_graph_drone = 0.0
        running_loss = 0.0
        running_corrects = 0.0
        running_corrects2 = 0.0

        # 开始小的Iterate
        for data, data2 in dataloaders:

            inputs, labels, _ = data  # 卫星数据
            inputs, labels = Variable(inputs.cuda().detach()), Variable(labels.cuda().detach())

            inputs2, labels2, label_id_dr = data2  # 无人机数据 / street data
            inputs2, labels2 = Variable(inputs2.cuda().detach()), Variable(labels2.cuda().detach())

            now_batch_size, _, _, _ = inputs.shape
            if now_batch_size < opt.batchsize:
                continue
            optimizer.zero_grad()  # 梯度清零

            outputs, outputs2 = model(inputs, inputs2)  # 返回概率值和特征
            logits, ff = outputs
            logits2, ff2 = outputs2
            # --------------------------------------- 添加局部分支 -------------------------------------- #
            # outputs, outputs2, local1, local2, x_g, x_drone = model(inputs, inputs2, adj_norm, adj_norm_drone)  # 返回概率值和特征
            # logits, ff = outputs
            # logits2, ff2 = outputs2
            # logitslol, fflocal = local1
            # logitslocal2, fflocal2 = local2
            # --------------------------------------- 添加局部分支 -------------------------------------- #

            _, preds = torch.max(logits.data, 1)
            _, preds2 = torch.max(logits2.data, 1)
            cls_loss = criterion(logits, labels) + criterion(logits2, labels2)

            if isinstance(preds, list) and isinstance(preds2, list):
                print("yes")
            if opt.triplet:  # 是否使用三元组损失
                f_triplet_loss = cal_triplet_loss(ff, ff2, labels, labels2, triplet_loss)
                # --------------------------------------- 添加局部分支 -------------------------------------- #
                # trip_loss_local = cal_triplet_loss(fflocal, fflocal2, labels, labels2, triplet_loss)
                # f_triplet_loss = trip_loss_local*0.25 + f_triplet_loss
                # --------------------------------------- 添加局部分支 -------------------------------------- #

                # 加上域内的三元组损失  
                hard_pairs = miner(ff, labels)
                # hard_pairs2 = miner(ff2, labels2)

                f_triplet_loss = criterion_triplet(ff, labels,
                                                   hard_pairs) + f_triplet_loss  # + criterion_triplet(ff2, labels2, hard_pairs2)

                # f_triplet_loss = f_triplet_loss + cal_triplet_loss(ff, ff, labels, labels2, triplet_loss)

                loss = f_triplet_loss + cls_loss * 0.5
            else:
                loss = cls_loss

            if epoch < opt.warm_epoch:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up

            loss.backward()
            optimizer.step()

            # 记录数据
            if int(version[0]) > 0 or int(version[2]) > 3:
                running_loss += loss.item() * now_batch_size
            else:
                running_loss += loss.data[0] * now_batch_size

            running_cls_loss += cls_loss * now_batch_size
            if use_graph:
                running_graph += loss_G * now_batch_size
                running_graph_drone += loss_G_drone * now_batch_size
            if opt.triplet:
                running_triplet += f_triplet_loss * now_batch_size
            else:
                running_triplet = 0

            running_corrects += float(torch.sum(preds == labels.data))
            running_corrects2 += float(torch.sum(preds2 == labels2.data))

        scheduler.step()
        epoch_loss = running_loss / dataset_sizes
        epoch_cls_loss = running_cls_loss / dataset_sizes
        epoch_triplet_loss = running_triplet / dataset_sizes

        epoch_acc = running_corrects / dataset_sizes  # epoch 卫星正确率
        epoch_acc2 = running_corrects2 / dataset_sizes  # epoch 无人机正确率

        print(
            'Epoch: {}  Loss: {:.4f} Cls_Loss:{:.4f} Triplet_Loss {:.4f}  Satellite_Acc: {:.4f}  Drone_Acc: {:.4f}'.format(
                epoch, epoch_loss, epoch_cls_loss, epoch_triplet_loss, epoch_acc, epoch_acc2))  #
        if epoch_acc > 0.8 and epoch_acc2 > 0.6:
            if epoch_loss < MAX_LOSS and epoch > (num_epochs - 20) or epoch == num_epochs - 1:
                MAX_LOSS = epoch_loss
                save_network(model, opt.name, epoch)
                print(opt.name + " Epoch: " + str(epoch + 1) + " has saved with loss: " + str(epoch_loss))

        time_elapsed = time.time() - since
        print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    time_elapsed = time.time() - begin
    print('Total training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere
    # model = two_view_net(class_num=opt.nclasses, droprate=opt.droprate, stride=opt.stride, share_weight=opt.share,
    #                      circle=return_feature)

    # test dino
    model = VPRModel(
        # ---- Encoder
        backbone_arch='dinov2_vitl14',  # dinov2_vitb14 dinov2_vitl14 convnext
        use_graph=opt.use_graph,
        # backbone_config={
        #     'num_trainable_blocks': 4,
        #     'return_token': False,
        #     'norm_layer': True,
        # },
        agg_arch='Gem',
        agg_config={
            # 'num_channels': 1024,
            # 'num_clusters': 64,
            # 'cluster_dim': 128,
            # 'token_dim': 256,
            # 'channel': 1024,  # 1024 for AFIM+l  AFIM只需要这个参数
        },
        class_num=opt.nclasses
    )

    if opt.use_graph:
        print('use graph')
    else:
        print('did not use graph')
    # optimizer_ft = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=6e-5,
    #     weight_decay=9.5e-9
    # )
    #
    # exp_lr_scheduler = lr_scheduler.LinearLR(
    #     optimizer_ft,
    #     start_factor=1,
    #     end_factor=0.2,
    #     total_iters=4000
    # )
    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ignored_params = list(map(id, model.classifier.parameters()))  # 全连接层和其他的层是不一样的学习率
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    #
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.3 * opt.lr},  # 0.1
        {'params': model.classifier.parameters(), 'lr': opt.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    #
    criterion = nn.CrossEntropyLoss()
    # if opt.labelsmooth == 1:
    #     print("use label smooth")
    #     criterion = cross_entropy_loss()
    #
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=opt.steps, gamma=0.1)

    dir_name = os.path.join('checkpoints', opt.name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    copyfile('train.py', dir_name + '/train.py')
    copyfile('./model.py', dir_name + '/model.py')

    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    model = train_model(model.cuda(), criterion, optimizer_ft, exp_lr_scheduler, num_epochs=opt.num_epochs, use_graph=opt.use_graph)  # 调用训练函数
