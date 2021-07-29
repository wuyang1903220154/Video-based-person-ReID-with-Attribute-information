# encoding: utf-8
from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
import numpy as np
import torch
from sklearn.metrics import f1_score


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def disciminative(x):
    above_average = x >= np.min(x)
    r = np.zeros(above_average.shape[0])
    r[above_average] = 1 / np.sum(above_average)
    return r
# 如果是这个文件路径不存在的话，那么我们就是要创建这样一个文件路径的
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       计算和存储当前值和平均值的过程的。
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()  # 这里是来对相应的值的进行初始化的过程的。

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # 对这些值来进行更新的过程的。
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AttributesMeter(object):
    """Computes and stores the average and current value.
        是用来计算和存储当前值的过程的。
        len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1])

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    ## 开始的时候来对一些值进行初始化的过程的。
    def __init__(self, attr_num):
        self.attr_num = attr_num
        self.preds =  [[] for _ in range(attr_num)]
        self.gts = [[] for _ in range(attr_num)]
        self.acces = np.array([0 for _ in range(attr_num)])
        self.acces_avg = None
        self.f1_score_macros = None
        self.count = 0

    ## n = 1  这里是来对相应的值进行更新的过程的。
    def update(self, preds, gts, acces, n):
        self.count += n
        self.acces += acces
        for i in range(len(preds)):
            self.preds[i].append(preds[i])
            self.gts[i].append(gts[i])

    ## 得到相应的特征和对应的属性的值的过程的。
    def get_f1_and_acc(self, mean_indexes=None):
        if mean_indexes is None:
            mean_indexes = [_ for _ in range(self.attr_num)]
        if self.acces_avg is None:
            self.acces_avg = self.acces / self.count
        if self.f1_score_macros is None:
            self.f1_score_macros = np.array([f1_score(y_pred=self.preds[i], y_true=self.gts[i], average='macro') for i in [0, 1] + list(range(self.attr_num))])

        return self.f1_score_macros, self.acces_avg, np.mean(self.acces_avg[mean_indexes]), np.mean(self.f1_score_macros[mean_indexes])


## 这里是来对存储
def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    ## mkdir_if_missing判断这个文件夹是否是存在的，如果不存在，那么就是来创建新的文件夹的。
    mkdir_if_missing(osp.dirname(fpath))  ## dirname() 函数返回路径中的目录名称部分。
    torch.save(state, fpath)   ##  然后是来把状态信息存储在这个文件夹中的。
    if is_best:
        ## 这个是来实现文件备份的操作的过程的。
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))  ## 把文件从原路径，复制到目标路径下的文件当中的

## 写一个接口把控制台的训练或者是测试的文本进行输出的过程的。
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout  ## 以管道的方式对内容进行输出到文件当中的。
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))  ## 如果是不存在的时候，就来创建这样一个文件路径的
            self.file = open(fpath, 'w')  ## 以只读的方式打开文件的

    def __del__(self):
        self.close()      ## 退出操作

    def __enter__(self):
        pass    ## 通过操作

    def __exit__(self, *args):
        self.close()   ## 退出操作

    def write(self, msg):
        self.console.write(msg)  ## 以管道的方式写入到这个文件当中的。
        if self.file is not None:
            self.file.write(msg)   ## 把相应的内容写入到文件当中的。
    """
        flush来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
    """
    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            ##如果你准备操作一个Python文件对象f, 首先f.flush(),然后os.fsync(f.fileno()), 确保与f相关的所有内存都写入了硬盘.
            os.fsync(self.file.fileno())  ## 是强制的将文件写入到硬盘当中的。

    def close(self):
        self.console.close()  # 是对应的控制台的关闭的过程的。
        if self.file is not None:
            self.file.close()   ## 这里是来对文件进行关闭的操作的过程的。

def read_json(fpath):
    with open(fpath, 'r') as f:  ## 以只读方式来打开这个json文件
        obj = json.load(f)  ## 然后是来加载这个json文件的。
    return obj

def write_json(obj, fpath):  ## 往json文件中，写入数据
    mkdir_if_missing(osp.dirname(fpath))   ## 如果是不存在这个文件，就是要来创建这样一个文件的。
    with open(fpath, 'w') as f:    ## 以写的方式来打开这个文件
        ## json.dumps()函数是将一个Python数据类型列表表示往里面写入文件
        json.dump(obj, f, indent=4, separators=(',', ': '))


## 传入相应的模型和配置参数，传入相应优化器，和一些配置参数的过程的。
def make_optimizer(cfg, model):
    params = []  ## 这里是来定义一个参数列表
    for key, value in model.named_parameters():  ## 因为模型中是存在着很多的参数列表的，其中键和值，在这个模型参数当中的
        if not value.requires_grad:   ## 如果是对应的这个参数是没有梯度信息的，那么我们就是来进行下一次循环，这样选择下一个参数的
            continue
        lr = cfg.SOLVER.BASE_LR  ##  3e-4 这里是来对应的最开始的学习率的。
        ## 权重损失是来调节模型复杂度，用来对损失函数的影响的。
        weight_decay = cfg.SOLVER.WEIGHT_DECAY  ## 0.0005 设置最开始的权重损失的。权重损失是放在正则向前面的一个系数
        if "bias" in key:
            ## 3e-4  和SOLVER.BIAS_LR_FACTOR = 2 是对应的学习率的偏置
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR  # 其中的如果是偏置在这个关键字当中的话。
            ## SOLVER.WEIGHT_DECAY_BIAS = 0.这里是对应的权重衰减的偏置的
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    ## 这里是如果使用的优化器 SGD SOLVER.MOMENTUM = 0.9
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)  ## gatattr是来返回对象的属性值，这里是来使用的Adam算法的过程的。传入相应的优化器当中的
    return optimizer ## 这里是来返回相应的优化器


def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
