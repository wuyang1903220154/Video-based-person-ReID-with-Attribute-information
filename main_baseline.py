# encoding: utf-8
from __future__ import print_function, absolute_import  ## 加入这些包以后，这里是来对应的一些语法的格式，可以在pychon 3x上来进行执行的
import os    ## 对应的是导入操作系统所对应的命令
import sys  ##  系统所对应的命令的
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

from PIL import Image

# try:
import apex  ## 通过改变数据的格式来减少模型显存的占用的工具的。转换为float16来进行相应的计算的过程的。作用这样是可以来提升GPU的训练的效果的
from apex import amp
# except:
#     pass
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn  ## 这里是来进行相应的梯度进行更新的，加入这个以后是可以加快计算速度的。
from torch.utils.data import DataLoader  ## 这里是来吐数据的过程的。
from tqdm import tqdm   ## 这里是来对应的一些进度条的操作的过程的。

import data_manager
from lr_schedulers import WarmupMultiStepLR
from video_loader import VideoDataset   ## 这里是来对应的视频数据集的加载的过程的。
import transforms as T   ## 对所对应的数据集进行转换过程的。
import models  ## 导入相应的模型的。
from losses import CrossEntropyLabelSmooth, TripletLoss, TripletLossAttrWeightes, \
    CosineTripletLoss    ## 这里是来导入相应的损失函数的
from utils import AverageMeter, Logger, AttributesMeter, make_optimizer
from eval_metrics import evaluate_withoutrerank
from samplers import RandomIdentitySampler
import pandas as pd
from config import cfg   ## 这里还是对应项目的所定义的一些配置命令的参数的

## 是通过配置文件来调用配置参数，和来对配置参数进行修改的
parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser.add_argument(
    "--config_file", default="./configs/softmax_triplet_tlaw.yml", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)  # 这里是来修改配置文件所对应的选项的。

args_ = parser.parse_args()

## 如果是调用的配置文件不为空的时候。
if args_.config_file != "":
    cfg.merge_from_file(args_.config_file)  ## cfg参数引用的配置文件中存放的参数将默认值替换，函数的功能就是将cfg文件中的值替换config.py中的值
cfg.merge_from_list(args_.opts)  ##  是来进行对里面的值进行改变的。

tqdm_enable = False  # 这里是来对应的一些进度条信息的。

def main():
    runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  ## 按照这个格式来进行开始输出所对应的时间的信息的
    ## cfg.OUTPUT_DIR = "" 是用来存储训练的日志文件的。D:/datasets/logs/%Y-%m-%d_%H-%M-%S存储日志信息。
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, runId)  ## 把输出信息和时间信息来进行连接的过程的。
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)  ## 如果是不存在这样的路径，那么我们就是要创建这样的路径的。创建多级目录
    print(cfg.OUTPUT_DIR)   ## 这里是来打印出路径信息的。
    ## cfg.RANDOM_SEED = 999，这里可能是来设置随机种子
    torch.manual_seed(cfg.RANDOM_SEED)  ## 为cpu设置随机数的种子，这样就是每次能够生成固定的数，这样每次实验显示一致，就可以有利于实验的比较和改进
    random.seed(cfg.RANDOM_SEED)  ## 这里是也是来生成随机数的种子的。
    np.random.seed(cfg.RANDOM_SEED)   ## 其中是np也是要来生成随机数的种子的。
    ## cfg.MODEL.DEVICE_ID = 0 对于的是使用GPU所对应的数字的。
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    ## 如果是存在GPU的情况下就是使用GPU,cfg.MODEL.DEVICE == "cuda"那么就是来使用gpu进行训练的。
    use_gpu = torch.cuda.is_available() and cfg.MODEL.DEVICE == "cuda"
    ## EVALUATE_ONLY = True 如果不是评估的化，那么就是对相应输出路径输出训练日志信息，否则就是输出测试的日志信息的。
    if not cfg.EVALUATE_ONLY:
        sys.stdout = Logger(osp.join(cfg.OUTPUT_DIR, 'log_train.txt'))  # 这里如果是在训练阶段生成的日志信息的
    else:
        sys.stdout = Logger(osp.join(cfg.OUTPUT_DIR, 'log_test.txt'))  # 这里是在测试阶段的话，我们要来输出相应日志信息的。

    print("==========\nConfigs:{}\n==========".format(cfg))

    ## 如果是我们这里使用gpu的话，我们就是输出在第几个GPU上面的来进行运算的。
    if use_gpu:
        print("Currently using GPU {}".format(cfg.MODEL.DEVICE_ID))  # 如果是使用的gpu话，输出现在是使用的第一个gpu
        cudnn.benchmark = True   ##  这里是来加快运算速度的。
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)  ##为所有的gpu设置种子，然后是是来生成随机数。有利于实验改进和比较的过程的。
    else:
        print("Currently using CPU (GPU is highly recommended)")  ## 现在是使用cpu的强烈建议使用gpu的

    print("Initializing dataset {}".format(cfg.DATASETS.NAME))    ## 这里是来对应的是最开始所使用的数据集的比如有mars数据集和duke数据集

    ## 传入相应的数据集路径和对应的数据集所对应的名字的。然后就是来得到这个数据集
    dataset = data_manager.init_dataset(root=cfg.DATASETS.ROOT_DIR, name=cfg.DATASETS.NAME)
    ## cfg.ATTR_RECOG_ON  = true
    if cfg.ATTR_RECOG_ON:
        cfg.DATASETS.ATTR_LENS = dataset.attr_lens  ## 设置数据集的属性的长度的
    else:
        cfg.DATASETS.ATTR_LENS = []

    cfg.DATASETS.ATTR_COLUMNS = dataset.columns  ## 这里是来改变数据集所对应的行的
    print("Initializing model: {}".format(cfg.MODEL.NAME))  ## MODEL.NAME = 'resnet50'

    ## MODEL.ARCH = 'video_baseline' 如果是这两个是相等的话，那么就是要来进行下面的操作的过程的。
    if cfg.MODEL.ARCH == 'video_baseline':
        torch.backends.cudnn.benchmark = False   ## 这里是来不使用GPU的加速的过程的
        ## 然后这里是来对模型进行初始化的过程的。
        ## cfg.MODEL.ARCH = 'video_baseline' 这里是来使用基准线的网络的
        ## num_classes=dataset.num_train_pids 这里是来对应分类的所对应的类别数的
        ## pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE = imagenet 使用imagenet来对网络进行预训练或者是通过数据来对网络进行预训练的过程
        ## last_stride=cfg.MODEL.LAST_STRIDE = 1 这里是对于主干网络所对应的最后步长的。
        ## neck=cfg.MODEL.NECK 这里是对应的如果训练达到瓶颈的话，我们就使用bnneck
        ## model_name=cfg.MODEL.NAME = resnet50 这里是我们使用的基准线网络所对应的名字
        ## neck_feat=cfg.TEST.NECK_FEAT = after 这里是使用哪一个特征来进行相应的测试的过程的。
        ## model_path=cfg.MODEL.PRETRAIN_PATH 这里是对应的预训练过主干网络的所对应的模型的路径的
        ## fusion_method=cfg.MODEL.FUSION_METHOD = "ATTR_TA"  这里是对应的融合的方法的。
        ## attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS = "bce" 是对应的数据集属性的设置的
        model = models.init_model(name=cfg.MODEL.ARCH, num_classes=dataset.num_train_pids, pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
                                  last_stride=cfg.MODEL.LAST_STRIDE,
                                  neck=cfg.MODEL.NECK, model_name=cfg.MODEL.NAME, neck_feat=cfg.TEST.NECK_FEAT,
                                  model_path=cfg.MODEL.PRETRAIN_PATH, fusion_method=cfg.MODEL.FUSION_METHOD, attr_lens=cfg.DATASETS.ATTR_LENS, attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS)
    ##  这里是来对应的输出Model所对应尺寸的。也就是参数所对应的总体的量的。
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))  # 这里是来对应的模型的参数是多少的。

    ## 这里是来对训练数据集进行数据曾广的一些操作过程的。
    transform_train = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN),  ## 重新来定义图片所对应的尺寸的 [256, 128] 这里是在训练期间的
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB), ## 以0.5的概率来决定是否是要进行水平翻转的过程的。
        T.Pad(cfg.INPUT.PADDING),  ## INPUT.PADDING = 10 这里是来对应的填充的尺寸的
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),  ## 然后这里是来对应的随机裁剪的
        T.ToTensor(),  ## 把图像转换为张量的
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  ## 对图像进行正则化和均值化，这样是能够来便于进行计算的
        T.RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN) ## INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406] 这里是来对像素的随机正则化后，然后在0.5的概率进行随机擦除
    ])

    ## 这里是来对测试集，进行数据增广的过程的。
    transform_test = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),  ## [384, 128]
        T.ToTensor(),  ## 转化为张量
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ## 然后是来进行正则化的操作的过程的。
    ])

    # pin_memory = True if use_gpu else False  如果是使用gpu = true
    pin_memory = False  # 同步gpu和cpu的运算这样是能够加快运算效果的。

    ## 这里是来对应的加载的线程是 DATALOADER.NUM_WORKERS = 8
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.DATASETS.ATTR_COLUMNS = dataset.columns  ## 然后是来数据集，属性是对应的有哪些数据类型的。

    ## 然后是这里开始是对训练数据集的加载的过程的。
    ## dataset.train 这里是来加载的训练的数据集的
    ## seq_len=cfg.DATASETS.SEQ_LEN = 4 是对应的每个训练或者是图像序列所对应的长度的
    ## sample=cfg.DATASETS.TRAIN_SAMPLE_METHOD 这里是对应的序列的采样的方法的
    ## 从视频中选择为1*SEQ_LEN 从一个序列视频中来选择4张图像的
    ## transform=transform_train 然后是来对视频进行上面的一些转换操作的过程的
    ## attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS ="bce" 属性损失是采用bce的属性损失的过程的。
    ## attr_lens=cfg.DATASETS.ATTR_LENS  ？
    ## dataset_name=cfg.DATASETS.NAME  这里是来对应数据集所对应的名称
    ## num_instances=cfg.DATALOADER.NUM_INSTANCE = 16 这里是对应每个批次所对应的实例数的
    ##
    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=cfg.DATASETS.SEQ_LEN, sample=cfg.DATASETS.TRAIN_SAMPLE_METHOD, transform=transform_train,
                     attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS, attr_lens=cfg.DATASETS.ATTR_LENS, dataset_name=cfg.DATASETS.NAME),
        ## 这里是定义的自定义的随机采样的 随机采样n个id，对于每一个id，随机采样 k个实例，所以 batch size的大小为n*k
        sampler=RandomIdentitySampler(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE),
        ## SOLVER.SEQS_PER_BATCH = 64 这里是对应的每个批次所对应的图像的数量的。
        ## DATALOADER.NUM_WORKERS = 8 这里是对应数据所加载的线程的数量的
        ## pin_memory=pin_memory 如果是使用GPU的话就是为true
        ## drop_last = True 是对最后一个批次数据也就是不满一个批次数据进行丢掉的。
        batch_size=cfg.SOLVER.SEQS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=True
    )

    ## 这里是来对查询数据集进行加载的过程的
    ## dataset.query 加载查询数据集
    ## seq_len=cfg.DATASETS.SEQ_LEN = 4 对于每个训练或者是图像序列所对应的长度的
    ## sample=cfg.DATASETS.TEST_SAMPLE_METHOD 这里是来对应的序列的采样的方法的
    ## transform=transform_test 对测试数据集来进行一些转换的操作的过程的
    ## max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM = 200 测试的时候所对应的最大序列的长度
    ## dataset_name=cfg.DATASETS.NAME 这里是对应数据集所对应的名称
    ## attr_lens=cfg.DATASETS.ATTR_LENS ？
    ## attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS = “bce"这里是来对数据集的属性进行设置的
    ## cfg.TEST.SEQS_PER_BATCH = 128 这里是对应的测试序列的所对应的批次
    ## shuffle = False 是不来对相应的数据进行打乱的
    ## num_workers=cfg.DATALOADER.NUM_WORKERS = 8 这里是来工作的线程数的
    ## drop_last = False 不满最后一个批次数据是不丢掉的。
    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=cfg.DATASETS.SEQ_LEN, sample=cfg.DATASETS.TEST_SAMPLE_METHOD, transform=transform_test,
                     max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME, attr_lens=cfg.DATASETS.ATTR_LENS, attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS),
        batch_size=cfg.TEST.SEQS_PER_BATCH , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False
    )

    ## gallery图库数据集的加载的过程的。
    ## dataset.gallery这里是把gallery数据集个加载过来
    ## seq_len=cfg.DATASETS.SEQ_LEN = 4 对于每个训练或者是图像所对应的长度的
    ## sample=cfg.DATASETS.TEST_SAMPLE_METHOD 这里是对应的序列的采样的方法的
    ## transform=transform_test 这里是来对galley加载的数据来进行转换的过程的
    ## max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM = 200 测试的时候所对应的最大序列的长度
    ## dataset_name=cfg.DATASETS.NAME 是对应的数据集所对应的名字
    ## attr_lens=cfg.DATASETS.ATTR_LENS ？
    ## attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS = "bce" 这里是来对数据集进行属性设置的
    ## batch_size=cfg.TEST.SEQS_PER_BATCH = 128 每批数据是对应的128
    ## shffle = False 是不对数据集进行打乱
    ## num_workers=cfg.DATALOADER.NUM_WORKERS = 8 使用多个线程来进行计算的
    ## pin_memory=pin_memory  是否是使用GPU来进行计算的
    ## drop_last=False 对于最后不满一个batch数据是不进行丢掉的。
    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=cfg.DATASETS.SEQ_LEN, sample=cfg.DATASETS.TEST_SAMPLE_METHOD, transform=transform_test,
                     max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME, attr_lens=cfg.DATASETS.ATTR_LENS, attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS),
        batch_size=cfg.TEST.SEQS_PER_BATCH , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False,
    )

    ## MODEL.SYN_BN = True
    if cfg.MODEL.SYN_BN:
        ## 这里是来让模型使用分布式的计算的方式的。
        # Apex为我们实现了同步BN，用于解决单GPU的minibatch太小导致BN在训练时不收敛的问题。
        # from apex.parallel import DistributedDataParallel
        model = apex.parallel.convert_syncbn_model(model) ## 把模型放在apex上面来进行计算这样是可以来减少计算量的。








    ##  这里是对应优化器，把参数和model传入后
    optimizer = make_optimizer(cfg, model)

    ## cfg.SOLVER.FP_16 = True的时候
    if cfg.SOLVER.FP_16:  ## 我们是来把相应的模型的初始化的过程的就是放在这上面来进行计算的过程，这样是可以来减少相应的计算量的。
        # O1表示的是混合精度训练的过程的。
        model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level='O1')
    ## 这里是来模型使用GPU来进行相应的计算的过程的。
    if use_gpu:
        model = nn.DataParallel(model)  ## 这里是让模型分为几个块来进行相应的计算的过程的。这里是来采样分布式计算的方式的。
        model.cuda()     ## 这里是让模型来使用GPU计算的过程的。
    ## 这里是模型的初始化已经完成了，下面就是正式的开始计算的过程了。
    ## 设置相应的开始的时间的
    start_time = time.time()
    ## 数据集所对应的训练的标签，然后是来对应相应的交叉熵的损失函数的过程的。这里是对应的标签平滑的交叉熵的损失的
    ## 这样是可以抑制正负样本的输出差值，使得网络是有更好泛化的能力的。
    xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids)

    # 这里是等于的DISTANCE == "cosine"是满足条件的这里就是能够使用余弦的三元组损失的过程的。
    if cfg.DISTANCE == "cosine":
        tent = CosineTripletLoss(cfg.SOLVER.MARGIN)  ## SOLVER.MARGIN = 0.3这里是来对应的定义三元组损失所对应的边界的。
    elif cfg.DISTANCE == "euclid":   ## 这里是来对应的欧式距离所对应的三元组的损失的，这里是不进行使用的。
        tent = TripletLoss(cfg.SOLVER.MARGIN)



    if cfg.DATASETS.ATTRIBUTE_LOSS == "mce":  ## 这里是来对应的数据集属性的损失
        attr_criter = nn.CrossEntropyLoss()  # 这里是对应属性的交叉熵损失的过程d
    elif cfg.DATASETS.ATTRIBUTE_LOSS == "bce":  ## 这里是对应的BCE的交叉熵的损失的
        # https://blog.csdn.net/qq_22210253/article/details/85222093 对于BCE损失
        attr_criter = nn.BCEWithLogitsLoss()  # 其中所对应BCE损失是使用sigmoid把所有的数据归一化到0-1之间，然后是在来计算相应BCE_loss的过程的


    ## 这个损失就是使用AITL来解决类内的图片帧很大的问题的。 # 这里是使用的是余弦距离的。
    # tlaw = TripletLossAttrWeightes(dis_type=cfg.DISTANCE)


    ## optimizer  这里是来把优化器进行加入的
    ## scheduler 翻译为调度程序
    ## cfg.SOLVER.STEPS =  (30, 55)这里是对应的学习率的损失的步骤的 当epoch = 30的时候学习率发生变化 当epoch =55的学习率也是会发生衰减的。
    ## cfg.SOLVER.GAMMA = 0.1 这里是来对应学习率的衰减率的
    ## cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3 这里是对应的预热系数
    ## cfg.SOLVER.WARMUP_ITERS = = 500 预测的迭代的步骤的
    ## cfg.SOLVER.WARMUP_METHOD = "linear" 这里是预测的方法是选择的线性的方法的
    ## WarmupMultiStepLR() 这里是来对应的学习率预热的方法的。训练一定的epochs后来修改学习率，在小的学习率下，模型是可以慢慢的趋于稳定的
    ## 等模型趋于稳定后在来选择预先设置的学习率进行训练，使得模型收敛速度更加的快，模型的效果更加。
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    ema = None
    no_rise = 0
    metrics = test(model, queryloader, galleryloader, cfg.TEST.TEMPORAL_POOL_METHOD, use_gpu)
    # return
    best_rank1 = 0
    start_epoch = 0
    # cfg.SOLVER.MAX_EPOCHS = 50
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        # if no_rise == 10:
        #     break
        scheduler.step()
        print("noriase:", no_rise)
        print("==> Epoch {}/{}".format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))  # 输出相应的epoch的过程的。
        print("current lr:", scheduler.get_lr()[0])  # 输出现在的对应的学习率的过程的。

        # 也就是是tlaw = tlaw这个属性属性损失给去掉的。
        #train(model, trainloader, xent, tent, attr_criter, optimizer, use_gpu, tlaw=tlaw)  # 这里是把AITL这个属性损失给去掉的
        train(model, trainloader, xent, tent, attr_criter, optimizer, use_gpu)
        # cfg.SOLVER.EVAL_PERIOD = 50
        # 满足下面的条件，然后就是进入测试的过程的。
        if cfg.SOLVER.EVAL_PERIOD > 0 and ((epoch + 1) % cfg.SOLVER.EVAL_PERIOD == 0 or (epoch + 1) == cfg.SOLVER.MAX_EPOCHS):
            print("==> Test")

            metrics = test(model, queryloader, galleryloader, cfg.TEST.TEMPORAL_POOL_METHOD, use_gpu)
            rank1 = metrics[0]  # 其中这里是来获得rank-1的值。
            if rank1 > best_rank1:  # rank1>best_rank1的值，那么我们就是要对rank1的值的进行更新的过程的。
                best_rank1 = rank1
                no_rise = 0
            else:
                no_rise += 1
                continue

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 然后是把这些model的参数给存储起来的。
            torch.save(state_dict, osp.join(cfg.OUTPUT_DIR, "rank1_" + str(rank1) + '_checkpoint_ep' + str(epoch + 1) + '.pth'))

    elapsed = round(time.time() - start_time)  # 这里是训练+数据加载总共的花的时间。
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, trainloader, xent, tent, attr_criter, optimizer, use_gpu, tlaw=None):
    model.train()   # 这里是model的训练的过程的。
    xent_losses = AverageMeter()  # 对应的交叉熵损失的过程的。
    tent_losses = AverageMeter()  # 三元组损失
    losses = AverageMeter()  # 总的损失
    attr_losses = AverageMeter()  # 属性损失
    tlaw_losses_unrelated = AverageMeter() # 是对应的损失的更新的过程的。这里是对应id不相关的。AITL属性损失更新的过程的。
    tlaw_losses_related = AverageMeter() # 这里是对应的id相关的。

    # [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0] = attrs

    for batch_idx, (imgs, pids, _, attrs) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()  # 把图片和id放在gpu上面来进行计算的。
            if cfg.DATASETS.ATTRIBUTE_LOSS == "mce":
                attrs = [a.view(-1).cuda() for a in attrs]
            else:
                attrs = [a.cuda() for a in attrs]  # 把相应属性放到cuda上面来进行计算的过程的。
        outputs, features, attr_preds = model(imgs) # 输出相应的特征，标签，和attr_preds预测
        # combine hard triplet loss with cross entropy loss
        xent_loss = xent(outputs, pids)  # 其中是预测的标签，与真实的标签来计算出相应损失的过程的。
        tent_loss, _, _ = tent(features, pids) # 这里是来对应的三元组损失的过程的。
        xent_losses.update(xent_loss.item(), 1) # 是来对损失进行更新的过程的。
        tent_losses.update(tent_loss.item(), 1)

        if cfg.ATTR_RECOG_ON:
            attr_loss = attr_criter(attr_preds[0], attrs[0])  # 计算出属性损失的过程的。是通过属性预测标签和真实标签之间的差距的过程的。
            for i in range(1, len(attrs)):
                attr_loss += attr_criter(attr_preds[i], attrs[i])  # 把相应的属性损失给叠加起来。
            if cfg.DATASETS.ATTRIBUTE_LOSS == "mce":
                attr_loss /= len(attr_preds)
            attr_losses.update(attr_loss.item(), 1) # 这里是来对属性损失的更新的过程的。
            # loss = xent_loss + tent_loss
            if cfg.DATASETS.ATTRIBUTE_LOSS == "mce":
                unrelated_attrs = torch.cat(attr_preds[:len(cfg.DATASETS.ATTR_LENS[0])], 1)
                related_attrs = torch.cat(attr_preds[len(cfg.DATASETS.ATTR_LENS[0]) : len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1])], 1)
            if cfg.DATASETS.ATTRIBUTE_LOSS == "bce":
                unrelated_attrs = attr_preds[0]
                related_attrs = attr_preds[1]

            # 这个损失是AITL来解决同一个id下DVDP问题的
            # tlaw_loss_unrelated = tlaw(features, pids, unrelated_attrs)
            # tlaw_loss_related = tlaw(features, pids, related_attrs)
            # tlaw_losses_unrelated.update(tlaw_loss_unrelated.item(), 1)
            # tlaw_losses_related.update(tlaw_loss_related.item(), 1)
            # tent_loss = (tent_loss + tlaw_loss_unrelated + tlaw_loss_related) / 3
            loss = xent_loss + tent_loss  # 然后是来对这两个损失就进行相加的过程的。

            loss += attr_loss  # 这里在来加上相应的属性损失的。

            optimizer.zero_grad()  # 这里是对应优化器里面的提度全部为0的情况的。

            if cfg.SOLVER.FP_16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()  # 是来对应的梯度反传的。
            else:
                loss.backward()

            optimizer.step()
            # ema.update()

        # 这里是不存在属性损失的情况的。
        else:
            loss = xent_loss + tent_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ema.update()
        losses.update(loss.item(), 1)  # 最后是对损失进行更新的过程的。


        # attr_losses.update(attr_loss.item(), pids.size(0))
    # print("Batch {}/{}\t Loss {:.6f} ({:.6f}), attr Loss {:.6f} ({:.6f}), related attr triplet Loss {:.6f} ({:.6f}), unrelated attr triplet Loss {:.6f} ({:.6f}),  xent Loss {:.6f} ({:.6f}), tent Loss {:.6f} ({:.6f})".format(
    #     batch_idx + 1, len(trainloader), losses.val, losses.avg, attr_losses.val, attr_losses.avg,
    #     tlaw_losses_unrelated.val, tlaw_losses_unrelated.avg, tlaw_losses_related.val, tlaw_losses_related.avg, xent_losses.val, xent_losses.avg,
    #     tent_losses.val, tent_losses.avg))
    print(
        "Batch {}/{}\t Loss {:.6f} ({:.6f}), attr Loss {:.6f} ({:.6f}),  xent Loss {:.6f} ({:.6f}), tent Loss {:.6f} ({:.6f})".format(
            batch_idx + 1, len(trainloader), losses.val, losses.avg, attr_losses.val, attr_losses.avg,
            xent_losses.val, xent_losses.avg,tent_losses.val, tent_losses.avg))


    return losses.avg

## 验证数据集：传入训练好的模型的，queryloader是对应的测试数据集，galleryloader是对应的图库数据集
## pool 使用相应的池化的操作的，use_gpu：使用gpu来进行计算，ranks我们所使用rank排名的。
def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    temp_recog_on = False
    if cfg.DATASETS.ATTRIBUTE_LOSS == "bce":   ## 如果是数据集的属性损失是等于bce，测试的时候就是采用 BCEWithLogitsLoss()
        temp_recog_on = cfg.ATTR_RECOG_ON  ## ATTR_RECOG_ON = True
        cfg.ATTR_RECOG_ON = False   ## 然后这里是来改变相应的值为false

    ## cfg.ATTR_RECOG_ON = true 我们来进行下面的操作的。因为是等于false所以是不用进行attr_metrics的操作的过程的。
    if cfg.ATTR_RECOG_ON:
        ## 我们是来进行属性距离的度量的。
        attr_metrics = AttributesMeter(len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1]))
    ## 在测试的时候是没有梯度进行更新的。
    with torch.no_grad():
        model.eval()
        qf, q_pids, q_camids = [], [], []   ## 是来得到query的特征，query的id号，query的相机id号的。
        ## 是对应的批次id号的，tqdm()是以进度条的方式来进行加载的过程的。
        for batch_idx, (imgs, pids, camids, attrs, img_path) in enumerate(tqdm(queryloader)):

            if use_gpu:
                imgs = imgs.cuda()   ## 对于图片的运算是使用gpu来进行相应的计算的。
                attrs = [a.view(-1) for a in attrs]  ## 是view(-1)是通过相应的属性的来自动调整维度的。
            b, n, s, c, h, w = imgs.size()  ## 得到图像序列所对应的维度的。
            assert (b == 1)   ## 判断b是否等于1
            imgs = imgs.view(b * n, s, c, h, w)  ##  这里是的来重新改变图像序列的所对应的尺寸的。
            ## 对于这里是outputs的值是输出的什么东西的
            features, outputs = model(imgs)  ## 然后是把相应的图像序列输入的后，得到相应的特征序列后，
            q_pids.extend(pids) ## extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。这样就是能够把全部的行人id加入到列表当中的。
            q_camids.extend(camids)  ## 这里也是把全部的相机号加入到列表当中的。

            features = features.view(n, -1)   ## 这里也是来转变特征对应的相应的维度的。

            features = torch.mean(features, 0)  ##  是按照z方向上的维度来来求取相应的平均值的。
            qf.append(features) ## 把得到的特征序列添加到查询的列表当中的。
            ## ATTR_RECOG_ON = true
            if cfg.ATTR_RECOG_ON:
                ## 对于是经过网络所对应的输出的值，dim = 0的方向上来求出相应的均值，并且，是把维度降维一维的
                outputs = [torch.mean(out, 0).view(1, -1) for out in outputs]
                preds = []
                gts = []
                ## range 是用来创建一个列表()然后用到for循环中的。
                acces = np.array([0 for _ in range(len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1]))])
                ## 是对于每个outputs进行输出后
                for i in range(len(outputs)):
                    outs = outputs[i].cpu().numpy()  ## 然后是来通过cpu转换numpy()的形式，然后hi在来进行处理的
                    # outs = torch.mean(outs, 0)  ## 转换后来求出相应的dim = 0 上来求出均值的
                    if cfg.DATASETS.ATTRIBUTE_LOSS == "ce":  ## 这里可能应该"bce"的值，如果是相等的话。
                        preds.append(np.argmax(outs, 1)[0])  ## 找出相应的最大值，然后是返回相应的最大值，所对应的位置的。
                        gts.append(attrs[i].cpu().numpy()[0])  ## 然后是把在来对相应属性进行转换的过程的。
                        acces[i] += np.sum(np.argmax(outs, 1) == attrs[i].numpy())  ## 如果是两个值是相等的话，那么就是来进行添加的
                attr_metrics.update(preds, gts, acces, 1)  ## 是来属性距离的度量后，然后来进行更新的过程的。
            del imgs  ## 这里是来释放图像所对应的引用的变量的。
            del outputs
            del features
        qf = torch.stack(qf)  ##  是对查询图像的维度上的拼接的过程的。这样是可以通过两维上的数据然后是变化为3维度上所对应的数据的。
        q_pids = np.asarray(q_pids) ## 这个是来对数据类型进行转换的。这里是来将列表转换为数组所对应的形式的。
        q_camids = np.asarray(q_camids)  ##这里也是转换相应数组所对应的形式的。

        ## 用来对特征数据集萃取相应的特征的。
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []  ## 对于图像库中的gf特征，和相应的行人的id，和和相应的camid号的。
        gallery_pathes = []  ## 这里是来对应的gallery的路径的
        for batch_idx, (imgs, pids, camids, attrs, img_path) in enumerate(tqdm(galleryloader)):
            # if batch_idx > 10:
            #     break
            gallery_pathes.append(img_path[0])   ## 首先是来把图像的路径的添加到里面
            if use_gpu:    ## 如果是存在gpu的话，那就是使用gpu来进行计算的
                imgs = imgs.cuda()
                attrs = [a.view(-1) for a in attrs]   ## 就是通过属性的来自动的调整相应的维度的。
            b, n, s, c, h, w = imgs.size()  ## 来得到相应的图像的所对应的尺寸的
            imgs = imgs.view(b * n, s, c, h, w)  ## 这里是来对图像的，的尺寸进行改变的。
            assert (b == 1)    ## 生成相应的测试功能，就是对这段话进行测试的意思的。
            features, outputs = model(imgs)  ## 通过模型来得到相应的特征的，和相应的输出值的过程的。

            features = features.view(n, -1)   ## 然后是来对特征的维度进行转换的过程的。
            if pool == 'avg':
                features = torch.mean(features, 0)   ## 这里是来进行平均池化的操作的
            else:
                features, _ = torch.max(features, 0)  ## 这里是来进行最大池化的操作的。
            g_pids.extend(pids)   ## 把相应的id添加到列表的当中的。
            g_camids.extend(camids)  ##  相应的相机的添加到列表当中的。
            gf.append(features)  ## 把特征的来添加到库里面的
            ## DATASETS.ATTRIBUTE_LOSS = "bce"
            if cfg.ATTR_RECOG_ON and len(attrs) != 0:
                ## 对于是经过网络所对应的输出的值，dim = 0的方向上来求出相应的均值，并且，是把维度降维一维的
                outputs = [torch.mean(out, 0).view(1, -1) for out in outputs]
                preds = []
                gts = []
                acces = np.array([0 for _ in range(len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1]))])
                for i in range(len(outputs)):
                    outs = outputs[i].cpu().numpy()  ## 转换成为numpy()然后是放在cpu()来进行计算的。
                    # outs = torch.mean(outs, 0)
                    if cfg.DATASETS.ATTRIBUTE_LOSS == "ce":
                        preds.append(np.argmax(outs, 1)[0])
                        gts.append(attrs[i].cpu().numpy()[0])
                        acces[i] += np.sum(np.argmax(outs, 1) == attrs[i].numpy())
                attr_metrics.update(preds, gts, acces, 1)
            del imgs   ## 这里是来对内存进行释放掉的。
            del outputs
            del features

        gf = torch.stack(gf) ##  是对库图像的维度上的拼接的过程的。这样是可以通过两维上的数据然后是变化为3维度上所对应的数据的。
        # gf = torch.stack(gf)
        g_pids = np.asarray(g_pids)   ## 这里都是来转化为数组所对应的形式的。
        g_camids = np.asarray(g_camids)

        ## 然后是来萃取特征对于gallery数据集的。这样是能够得到gallery的特征的。
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")  ## 然后我们这里来开始，计算距离矩阵的。

        ## 这里是如果数据集是gallery数据集化，那么我们就是能够
        if cfg.DATASETS.NAME == "duke":
            print("gallary with query result:")  ## 答应出query的查询的结果的。
            gf = torch.cat([gf, qf], 0)  ## 然后是对这两个特征的进行连接的。
            g_pids = np.concatenate([g_pids, q_pids], 0)  ## 这里是在维度上进行拼接的函数的。这里是按照行数，一行一行的拼接的
            g_camids = np.concatenate([g_camids, q_camids], 0)  ## 这里也是对应的拼接的函数的。
            ## 使用的相应的余弦函数来进行度量，相应的距离的。
            metrics = evaluate_withoutrerank(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks, dis_type=cfg.DISTANCE)
        else:
            ## 这里是为mars数据集来对应的所对应的操作的过程的。也就是来求出相应的余弦距离的。
            metrics = evaluate_withoutrerank(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks, dis_type=cfg.DISTANCE)
        ## cfg.ATTR_RECOG_ON = true
        if cfg.ATTR_RECOG_ON:

            print("Attributes:")  ## 这里是来对应的属性的。
            print("single performance:")  ## 单一性能的
            f1_score_macros, acces_avg, acc_mean_all, f1_mean_all = attr_metrics.get_f1_and_acc()
            colum_str = "|".join(["%15s" % c for c in cfg.DATASETS.ATTR_COLUMNS])
            acc_str = "|".join(["%15f" % acc for acc in acces_avg])
            f1_scores_macros_str = "|".join(["%15f" % f for f in f1_score_macros])
            print(colum_str)
            print(acc_str)
            print(f1_scores_macros_str)

            mean_columns = ["mean_all", "mean_related", "mean_no_quality"]
            mean_acces = [acc_mean_all, np.mean(acces_avg[range(len(cfg.DATASETS.ATTR_LENS[0]), len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1]))]), \
                          np.mean(acces_avg[[0, 1] + list(range(len(cfg.DATASETS.ATTR_LENS[0]), len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1])))])]
            mean_f1s = [f1_mean_all, np.mean(f1_score_macros[range(len(cfg.DATASETS.ATTR_LENS[0]), len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1]))]), \
                          np.mean(f1_score_macros[[0, 1] + list(range(len(cfg.DATASETS.ATTR_LENS[0]), len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1])))])]

            colum_str = "|".join(["%15s" % c for c in mean_columns])
            acc_str = "|".join(["%15f" % acc for acc in mean_acces])
            f1_scores_macros_str = "|".join(["%15f" % f for f in mean_f1s])
            print(colum_str)
            print(acc_str)
            print(f1_scores_macros_str)

        cfg.ATTR_RECOG_ON = temp_recog_on
        return metrics


if __name__ == '__main__':

    main()   ## 然后是程序从这里开始来进行执行的。




