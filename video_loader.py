# encoding: utf-8
from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
import random
from IPython import embed

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).  # 这里是来对应的batch数据集，和对应的batch,
    """
    sample_methods = ['evenly', 'random', 'all']  # 对于视频数据集，是有3种采样的方式的

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, attr_loss="mce", attr_lens=[], max_seq_len=200, dataset_name="mars"):
        self.dataset = dataset  # 首先是来把训练数据集给传输过来。
        self.seq_len = seq_len # seq_len = 4 对于每个视频序列是从中来提取4张图片的。
        self.sample = sample  # 传输过来的采样方法是 random 采样的方法的。
        self.transform = transform # 对训练数据集，进行数据增广的过程的。
        self.attr_loss = attr_loss  # 属性损失采用 bce损失的过程的。
        self.attr_lens = attr_lens  # 传入的是属性数据集的标签信息的。
        self.max_seq_len = max_seq_len # 其中最大序列长度是200
        self.dataset_name = dataset_name # mars数据集
        self.attr_lens = attr_lens
    def __len__(self):
        return len(self.dataset) # 这里在得到训练数据集的长度是8298

    def __getitem__(self, index):
        img_paths, pid, camid, attrs = self.dataset[index] # 是 读取这个轨道的所有图片。
        num = len(img_paths)
        attributes = []
        if self.attr_loss == "mce" and len(self.attr_lens) != 0:
            for a in attrs:
                attributes.append(Tensor([a]).long())

        # 我们这里采用的是bce这种损失函数的 是满足条件然后是来进行下面的操作的过程的。
        elif self.attr_loss == "bce" and len(self.attr_lens) != 0:
            attribute = []
            for i, a in enumerate(attrs):
                if i < len(self.attr_lens[0]):
                    attr = [1 if _ == a else 0 for _ in range(self.attr_lens[0][i])]
                    attribute.extend(attr)
                else:
                    attr = [1 if _ == a else 0 for _ in range(self.attr_lens[1][i - len(self.attr_lens[0])])]
                    attribute.extend(attr)
                if i == len(self.attr_lens[0]) - 1:
                    attribute = torch.Tensor(attribute)
                    attributes.append(attribute)
                    attribute = []

            attribute = torch.Tensor(attribute)
            attributes.append(attribute)
            # attributes = torch.Tensor([attributes]).long()

        # 我们是使用这一种方法来进行随机采样的过程的。
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.  # 如果是序列相比较是更小相比较seq_len，那么就是要重复进行的
            This sampling strategy is used in training phase. # 这个采样序列是被使用在训练阶段
            """
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid, attributes

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = list(range(num))
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)

                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            if len(imgs_list) > self.max_seq_len:
                sp = int(random.random() * (len(imgs_list) - self.max_seq_len))
                ep = sp + self.max_seq_len
                imgs_list = imgs_list[sp:ep]
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid, attributes, img_paths[0]

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))



if __name__ == '__main__':
    import data_manager  # 导入对数据集处理的文件
    dataset = data_manager.init_dataset(root='D:/datasets/mars',name = 'mars') # 这里是来对mars数据集进行处理的过程的。
    train_loader = VideoDataset(dataset.train) # market1501 的 train的数据集来进行加载的过程的。# train_loader这里来对应索引地址的
    print(train_loader.__len__())







