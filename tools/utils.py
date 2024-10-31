import os
import torch
import yaml
from torch.autograd import Variable

from model import two_view_net
import numpy as np
import random
import cv2

from tools.DefogFilter import GuidedFilter


# ------------------------------------------ Defog ----------------------------------------- #
# 用于排序时存储原来像素点位置的数据结构
class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def printInfo(self):
        print(self.x, self.y, self.value)


# 获取最小值矩阵
# 获取BGR三个通道的最小值
def getMinChannel(img):
    # 输入检查
    if len(img.shape) == 3 and img.shape[2] == 3:
        pass
    else:
        print("bad image shape, input must be color image")
        return None
    imgGray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMin = 255
            for k in range(0, 3):
                if img.item((i, j, k)) < localMin:
                    localMin = img.item((i, j, k))
            imgGray[i, j] = localMin
    return imgGray


# 获取暗通道
def getDarkChannel(img, blockSize):
    # 输入检查

    if len(img.shape) == 2:
        pass
    else:
        print("bad image shape, input image must be two demensions")
        return None

    # blockSize检查
    if blockSize % 2 == 0 or blockSize < 3:
        print('blockSize is not odd or too small')
        return None
    # print('blockSize', blockSize)
    # 计算addSize
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    # 中间结果
    imgMiddle = np.zeros((newHeight, newWidth))
    imgMiddle[:, :] = 255
    # print('imgMiddle',imgMiddle)
    # print('type(newHeight)',type(newHeight))
    # print('type(addSize)',type(addSize))
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle', imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    localMin = 255

    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMin = 255
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) < localMin:
                        localMin = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMin

    return imgDark


# 获取全局大气光强度
def getAtomsphericLight(darkChannel, img, meanMode=False, percent=0.001):
    size = darkChannel.shape[0] * darkChannel.shape[1]
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]

    nodes = []

    # 用一个链表结构(list)存储数据
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)

    # 排序
    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)

    atomsphericLight = 0

    # 原图像像素过少时，只考虑第一个像素点
    if int(percent * size) == 0:
        for i in range(0, 3):
            if img[nodes[0].x, nodes[0].y, i] > atomsphericLight:
                atomsphericLight = img[nodes[0].x, nodes[0].y, i]
        return atomsphericLight

    # 开启均值模式
    if meanMode:
        sum = 0
        for i in range(0, int(percent * size)):
            for j in range(0, 3):
                sum = sum + img[nodes[i].x, nodes[i].y, j]

        atomsphericLight = int(sum / (int(percent * size) * 3))
        return atomsphericLight

    # 获取暗通道前0.1%(percent)的位置的像素点在原图像中的最高亮度值
    for i in range(0, int(percent * size)):
        for j in range(0, 3):
            if img[nodes[i].x, nodes[i].y, j] > atomsphericLight:
                atomsphericLight = img[nodes[i].x, nodes[i].y, j]

    return atomsphericLight


# 恢复原图像
# Omega 去雾比例 参数
# t0 最小透射率值
def getRecoverScene(img, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001):
    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值

    imgGray = getMinChannel(img)
    # print('imgGray', imgGray)
    imgDark = getDarkChannel(imgGray, blockSize=blockSize)
    atomsphericLight = getAtomsphericLight(imgDark, img, meanMode=meanMode, percent=percent)

    imgDark = np.float64(imgDark)
    transmission = 1 - omega * imgDark / atomsphericLight

    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmission = guided_filter.filter(transmission)

    # 防止出现t小于0的情况
    # 对t限制最小值为0.1

    transmission = np.clip(transmission, t0, 0.9)

    sceneRadiance = np.zeros(img.shape)
    for i in range(0, 3):
        img = np.float64(img)
        sceneRadiance[:, :, i] = (img[:, :, i] - atomsphericLight) / transmission + atomsphericLight

    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    return transmission, sceneRadiance


# ------------------------------------------ Defog ----------------------------------------- #


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s' % dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


# Save model
def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./checkpoints/' + dirname):
        os.mkdir('./checkpoints/' + dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./checkpoints', dirname, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


#  Load model for resume
def load_network(name, opt):
    # Load config
    dirname = os.path.join('./checkpoints', name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if opt.epoch == 'last':
        if not epoch == 'last':
            epoch = int(epoch)
    else:
        epoch = opt.epoch
    config_path = os.path.join(dirname, 'opts.yaml')
    with open(config_path, 'r') as stream:
        # config = yaml.load(stream)
        config = yaml.safe_load(stream)

    opt.name = config['name']
    opt.data_dir = config['data_dir']
    opt.droprate = config['droprate']
    opt.color_jitter = config['color_jitter']
    opt.batchsize = config['batchsize']
    opt.h = config['h']
    opt.w = config['w']
    opt.share = config['share']
    opt.stride = config['stride']
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']
    if 'gpu_ids' in config:
        opt.gpu_ids = config['gpu_ids']
    opt.erasing_p = config['erasing_p']
    opt.lr = config['lr']
    opt.nclasses = config['nclasses']
    opt.erasing_p = config['erasing_p']

    model = two_view_net(opt.nclasses, opt.droprate, stride=opt.stride, share_weight=opt.share)

    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth' % epoch
    else:
        save_filename = 'net_%s.pth' % epoch

    save_path = os.path.join('./checkpoints', name, save_filename)
    print('Load the model from %s' % save_path)
    network = model
    network.load_state_dict(torch.load(save_path))
    return network, opt, epoch


def load_dino_network(opt, model):
    # Load config
    dirname = os.path.join('./checkpoints', opt.name)
    print(dirname)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if opt.epoch == 'last':
        if not epoch == 'last':
            epoch = int(epoch)
    else:
        epoch = opt.epoch

    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth' % epoch
    else:
        save_filename = 'net_%s.pth' % epoch

    save_path = os.path.join('./checkpoints', opt.name, save_filename)
    print('Load the model from %s' % save_path)
    network = model
    network.load_state_dict(torch.load(save_path))
    return network


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    toogle_grad(model_src, True)


# set 3407 for fun
def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def which_view(name):  # 根据视角返回参数 1 代表卫星 2 代表无人机
    if 'satellite' in name:
        return 1
    elif 'drone' in name:
        return 2
    else:
        print('unknown view')
    return -1


def fliplr(img):  # 将图像水平翻转
    """flip horizontal"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def get_id(img_path):
    labels = []
    paths = []
    for path, _ in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths


def extract_feature(model, dataloaders, view_index):  # 提取特征
    outputs = torch.FloatTensor()
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, _, _, _ = img.size()

        ff = torch.FloatTensor(n, 512).zero_().cuda()
        count += n
        print("已处理数据：", count)

        for i in range(2):  # 第一次对图像进行水平翻转处理 第二次不需要 最后输出的特征是两次之和
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())
            if view_index == 1:  # 1 代表卫星
                outputs, _ = model(input_img, None)
            elif view_index == 2:  # 2 代表无人机
                _, outputs = model(None, input_img)
            if i == 0:
                ff = outputs
            else:
                ff += outputs
        # 归一化
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        # 拼接
        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def extract_feature_dino_defog(model, dataloaders, view_index):  # 提取特征
    outputs = torch.FloatTensor()
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, img_de, label = data
        n, _, _, _ = img.size()

        ff = torch.FloatTensor(n, 512).zero_().cuda()
        count += n
        print("已处理数据：", count)

        for i in range(2):  # 第一次对图像进行水平翻转处理 第二次不需要 最后输出的特征是两次之和
            # if i == 2 and view_index == 2:
            # img = img_de

            if i == 1:
                # img = fliplr(img)
                img = img_de

            input_img = Variable(img.cuda())

            if view_index == 1:  # 1 代表卫星
                outputs, _ = model(input_img, None)
                # print(outputs.shape)
            elif view_index == 2:  # 2 代表无人机
                outputs, _ = model(input_img, None)
            if i == 0:
                ff = outputs
            else:
                ff += outputs
        # 归一化
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        # 拼接
        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def extract_feature_dino(model, dataloaders, view_index):  # 提取特征
    outputs = torch.FloatTensor()
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:

        img, label = data
        n, _, _, _ = img.size()

        ff = torch.FloatTensor(n, 512).zero_().cuda()
        count += n
        print("已处理数据：", count)

        for i in range(2):  # 第一次对图像进行水平翻转处理 第二次不需要 最后输出的特征是两次之和

            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())

            if view_index == 1:  # 1 代表卫星
                outputs, _ = model(input_img, None)
                # print(outputs.shape)
            elif view_index == 2:  # 2 代表无人机
                outputs, _ = model(input_img, None)
            if i == 0:
                ff = outputs
            else:
                ff += outputs
        # 归一化
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        # 拼接
        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def extract_feature_dino_no_defog(model, dataloaders, view_index):  # 提取特征
    outputs = torch.FloatTensor()
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:

        img, _, label = data
        n, _, _, _ = img.size()

        ff = torch.FloatTensor(n, 512).zero_().cuda()
        count += n
        print("已处理数据：", count)

        for i in range(2):  # 第一次对图像进行水平翻转处理 第二次不需要 最后输出的特征是两次之和

            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())

            if view_index == 1:  # 1 代表卫星
                outputs, _ = model(input_img, None)
                # print(outputs.shape)
            elif view_index == 2:  # 2 代表无人机
                outputs, _ = model(input_img, None)
            if i == 0:
                ff = outputs
            else:
                ff += outputs
        # 归一化
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        # 拼接
        features = torch.cat((features, ff.data.cpu()), 0)
    return features