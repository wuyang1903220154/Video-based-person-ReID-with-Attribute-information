# encoding: utf-8
from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import pandas as pd
import random
from collections import Counter

from tqdm import tqdm

from utils import mkdir_if_missing, write_json, read_json
from video_loader import read_image
import transforms as T
from IPython import embed
"""Dataset classes"""


class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    # 这里是对应的一个较大尺寸的数据集的。
    Dataset statistics:
    # identities: 1261  其中id身份是1261个
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)  # 其中所对应的视频序列的是全部这么多相加的过程的
    # cameras: 6  # 其中相机的6个

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
        是对应的长度小于min_seq_len的值将是会被丢弃的。
    """

    # 这里是来对应的属性的类型的。
    columns = ["action", "angle", "occulusion", "cut", "multi-person",
               "back-dominant", "target-change", "target-miss", "wrongtarget", "upcolor",
               "downcolor", "age", "up", "down", "bag",
               "backpack", "hat", "handbag", "hair",
               "gender", "btype"]
    attr_lens = [[5, 6, 2, 2, 2, 2, 2, 2, 2], [ 9, 10, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]]  # 这里是对每个属性所对应的值域的宽度的。

    # columns = ["action", "angle", "upcolor",
    #                       "downcolor", "age", "up", "down", "bag",
    #                       "backpack", "hat", "handbag", "hair",
    #                       "gender", "btype"]
    # attr_lens = [[5, 6], [9, 10, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
    def __init__(self, root, min_seq_len=0, attr=True):
        self.root = root
        self.train_name_path = osp.join(root, 'info/train_name.txt')
        self.test_name_path = osp.join(root, 'info/test_name.txt')  # 首先是来对原始路径进行连接的过程的。
        self.track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(root, 'info/query_IDX.mat')
        self.attributes_path = osp.join(root, "mars_attributes.csv")
        self._check_before_run()
        # prepare meta data
        train_names = self._get_names(self.train_name_path)   # 这里是来对应的全部训练集，所对应的图片的。 509914
        test_names = self._get_names(self.test_name_path)  # 681089
        self.attributes = pd.read_csv(self.attributes_path, encoding="UTF-8")  # 原本的编码是encoding = “gbk”  # 这里是来输出的相应属性数据集的列表的。
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)  [     1,     16,      1,      1]
        # 其中第一行和第二行代表着图片序号。第3列是行人id,第四列是对应的摄像头id
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4) [     1,     24,     -1,      1]
        # 其中这个列表分别是代表着，测试行人的图像的，和对应的想应的行人id和摄像头id的过程的。
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0  # 其中我们这里所对应的索引都是从0开始的
        track_query = track_test[query_IDX,:]  #  array([171610, 171649,      2,      1]) 这里是对应的需要用来查询的序列段的。
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]  # 这里来把query从测试数据集中，来进行排除的。
        track_gallery = track_test[gallery_IDX,:] # [     1,     24,     -1,      1] 这里是gallery中的视频的序列的。

        # num_train_tracklets总共对应的轨道数量
        # num_train_pids 训练的id数
        # num_train_imgs 训练的时候每个轨道所对应的图片的数量的
        # train ，训练图片每个tracklet所对应图片数，id,camid,
        #  'D:/datasets/mars\\bbox_train\\0091\\0091C1T0004F017.jpg',
        #    'D:/datasets/mars\\bbox_train\\0091\\0091C1T0004F020.jpg'),
        #   31, pid
        #   0, camid
        #   [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]) 是对应的属性集的。
        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len, attr=attr)
        # 'D:/datasets/mars\\bbox_test\\0646\\0646C2T0008F004.jpg',
        #    'D:/datasets/mars\\bbox_test\\0646\\0646C2T0008F012.jpg'),
        #   646,
        #   1,
        #   [0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 7, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1]),
        # num_query_tracklets = 1980
        # num_query_pids = 626
        # num_query_imgs是对应的每个序列的所对应的图片的数量的。
        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len, attr=attr)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len, attr=attr)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs  # 对应的全部的轨道序列的图片的
        min_num = np.min(num_imgs_per_tracklet)  # 这里是的对应的轨道序列中的最小值
        max_num = np.max(num_imgs_per_tracklet) # 轨道序列中的最大值920
        avg_num = np.mean(num_imgs_per_tracklet) # 轨道的平均值 59.54


        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        # 这里是来对这些结果进行展示后，然后来进行相应输出的过程的。
        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    # 核对这些路径是否是存在的。
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))
        if not osp.exists(self.attributes_path):
            raise RuntimeError("'{}' is not available".format(self.attributes_path))


    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    # 这里是来对数据集，进行处理的过程的
    # train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len, attr=attr
    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, attr=False):
        assert home_dir in ['bbox_train', 'bbox_test']  # 这里是来检查这里的训练数据是否是来自这两个数据集的。
        # 8298 这里是来训练的轨道的数量的。
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist())) # 使用集合是把里面重复的id给去除掉的。625 这里是总共是625个id对于训练集的。
        num_pids = len(pid_list)

        # 在训练的时候是要重置标签的。
        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}  # 这里是来把标签进行重置后，然后是从0开始的
        tracklets = []  # 定义相应的轨道序列的
        num_imgs_per_tracklet = []  # 对于每个序列的图片的数量的。
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored  # pid = -1的时候是对应垃圾数据的。
            img_names = names[start_index-1:end_index] # '0001C1T0001F001.jpg'对应的每个轨道序列的全部的图片的。
            attribute = []  # 这里是来对应的属性的列表的。
            # 对于这个attribute是没有怎么看懂的。
            if attr:
                t_id = int(img_names[0].split("F")[0].split("T")[1]) # '0001' 这里是把轨道号给提取出来的。
                attribute = self.attributes[(self.attributes.person_id == pid) & (self.attributes.camera_id == camid) & (
                        self.attributes.tracklets_id == t_id)].values
                if len(attribute) > 0:
                    attribute = attribute[0, 3:].tolist()
                    # attribute = attribute[0 : 2] + attribute[9 : ]
                    # attribute = attribute[2:9].tolist() + attribute[0:2].tolist() + attribute[9:].tolist()
                    # attributes.append(attribute)
                    # attribute = attribute[0, 12:]
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]  # 这里是来重新更新相应标签的。
            camid -= 1 # index starts from 0
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]  # 这里是把行人id给提取出来的。
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images" # 判断是否是单个序列中包含不同id的图像的

            # make sure all images are captured under the same camera* 确保图像被捕获在相同的摄像机下的。
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!" # 确保在同一个序列中的，摄像机的id是相同的。

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]

            # 这里是对应的在相同tracket-id下的最少图片数是大于min_seq_len的所对应的值的。
            if len(img_paths) >= min_seq_len:
                random.shuffle(img_paths)   # 就是随机来把这些图片进行打乱
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid, attribute))
                num_imgs_per_tracklet.append(len(img_paths))
        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet  # num_imgs_per_tracklet 每次每个轨道所对应的图片的数量的。

class iLIDSVID(object):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """
    root = './data/ilids-vid'
    dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
    data_dir = osp.join(root, 'i-LIDS-VID')
    split_dir = osp.join(root, 'train-test people splits')
    split_mat_path = osp.join(split_dir, 'train_test_splits_ilidsvid.mat')
    split_path = osp.join(root, 'splits.json')
    cam_1_path = osp.join(root, 'i-LIDS-VID/sequences/cam1')
    cam_2_path = osp.join(root, 'i-LIDS-VID/sequences/cam2')

    def __init__(self, split_id=0):
        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train

        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _download_data(self):
        if osp.exists(self.root):
            print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.URLopener()
        url_opener.retrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids/2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

class PRID(object):
    """
    PRID

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root = './data/prid2011'
    dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
    split_path = osp.join(root, 'splits_prid2011.json')
    cam_a_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_a')
    cam_b_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_b')

    def __init__(self, split_id=0, min_seq_len=0):
        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class DukeMTMC_Video(object):
    """
    DukeMTMC-vedio

    Reference:
    Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning. Wu et al., CVPR 2018

    Dataset statistics:
    702 identities (2,196 videos) for training and 702 identities (2,636 videos) for testing.
    # cameras: 8

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    # 这里是来总共的是对应多少个属性的
    columns = ["action", "angle", "occulusion", "cut", "multi-person",
               "back-dominant", "target-change", "target-miss", "wrongtarget",
               "backpack", "shoulder bag", "handbag", "boots", "gender", "hat", "shoes", "top", "downcolor", "topcolor"]
    attr_lens = [[5, 6, 2, 2, 2, 2, 2, 2, 2],[2, 2, 2, 2, 2, 2, 2, 2, 8, 9]]  # 每个属性所对应值域的宽度的，并且是前面是对应id不相关的属性后面是对应id相关的属性的

    # columns = ["action", "angle",
    #            "backpack", "shoulder bag", "handbag", "boots", "gender", "hat", "shoes", "top", "downcolor", "topcolor"]
    # attr_lens = [[5, 6], [2, 2, 2, 2, 2, 2, 2, 2, 8, 9]]

    def __init__(self, root, min_seq_len=0, attr=True):
        self.root = root
        self.train_name_path = os.path.join(root, "train")  # 这里是对应的训练图片所对应的路径
        self.gallery_name_path = os.path.join(root, "gallery") # 这里是对应的测试图片的所对应的路径的。gallery图库所对应的路径的。
        self.query_name_path = os.path.join(root, "query") # query图片所对应的路径的。
        self.attributes_path = osp.join(root, "duke_attributes.csv")  # 这里是来对应的属性数据集的过程的。
        self._check_before_run() # 用来检查这些数据集的路径是否是存在的。
        self.attributes = pd.read_csv(self.attributes_path) # 把属性数据集中的数据给读取出来 总共是有4832行 x 22 列数据，这样所对应22是对应有22个属性的。

        # 'E:/datasets/DukeMTMC-VideoReID\\train\\0640/1000\\0640_C1_F0087_X163321.jpg',
        #    'E:/datasets/DukeMTMC-VideoReID\\train\\0640/1000\\0640_C1_F0035_X163061.jpg',
        #    'E:/datasets/DukeMTMC-VideoReID\\train\\0640/1000\\0640_C1_F0025_X163011.jpg',
        #    'E:/datasets/DukeMTMC-VideoReID\\train\\0640/1000\\0640_C1_F0018_X162976.jpg',
        #    'E:/datasets/DukeMTMC-VideoReID\\train\\0640/1000\\0640_C1_F0023_X163001.jpg',
        #    'E:/datasets/DukeMTMC-VideoReID\\train\\0640/1000\\0640_C1_F0102_X163396.jpg',
        #    'E:/datasets/DukeMTMC-VideoReID\\train\\0640/1000\\0640_C1_F0124_X163506.jpg',
        #    'E:/datasets/DukeMTMC-VideoReID\\train\\0640/1000\\0640_C1_F0020_X162986.jpg'),
        #   467,
        #   0,
        #   [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
        # 这里是对应的train,里面所对应的图片路径，行人id号，相机id号，和所对应属性列表的过程的。
        # 2196 = num_train_tracklets
        # num_train_pids = 702
        # num_train_imgs # 对应训练图片的数量的
        # train_t_list # 训练所对应的全部的轨道号的。
        train, num_train_tracklets, num_train_pids, num_train_imgs, train_t_list = \
            self._process_data(self.train_name_path, relabel=True, min_seq_len=min_seq_len,
                               attr=attr)

        # 与上面的操作是一样的，但是relabel = False是没有标签重置的过程的
        query, num_query_tracklets, num_query_pids, num_query_imgs, query_t_list = \
            self._process_data(self.query_name_path, relabel=False, min_seq_len=min_seq_len,
                               attr=attr)


        # 与上面的操作是一样的，relabel = False是没有标签重置的过程的。
        # exclude_tracklets=query_t_list是对应的是要把查询的轨道进行排除的。
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, gallery_t_list = \
            self._process_data(self.gallery_name_path, relabel=False, min_seq_len=min_seq_len,
                               attr=attr, exclude_tracklets=query_t_list)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> Duke loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def get_mean_and_var(self):
        imgs = []
        for t in self.train:
            imgs.extend(t[0])
        channel = 3
        x_tot = np.zeros(channel)
        x2_tot = np.zeros(channel)
        for img in tqdm(imgs):
            x = T.ToTensor()(read_image(img)).view(3, -1)
            x_tot += x.mean(dim=1).numpy()
            x2_tot += (x ** 2).mean(dim=1).numpy()

        channel_avr = x_tot / len(imgs)
        channel_std = np.sqrt(x2_tot / len(imgs) - channel_avr ** 2)
        print(channel_avr, channel_std)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.gallery_name_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_name_path))
        if not osp.exists(self.query_name_path):
            raise RuntimeError("'{}' is not available".format(self.query_name_path))
        if not osp.exists(self.attributes_path):
            raise RuntimeError("'{}' is not available".format(self.attributes_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    # 这里就是来开始对数据进行处理的过程
    def _process_data(self, home_dir, relabel=False, min_seq_len=0, attr=False, exclude_tracklets=None):
        pid_list = []
        tracklets_path = []  # 定义相应的列表来存储相关的数据的
        tracklets_list = []
        # E:/datasets/DukeMTMC-VideoReID\\train是来对应的训练数据集所对应的路径的。
        # p 首先是来获取相关的文件号的0001也就是来对应的id号的。
        for p in os.listdir(home_dir):
            # 因为在同一个行人id下可能是有几个不同的序列id，然后在来对每个序列id进行处理的过程的。
            for t in os.listdir(os.path.join(home_dir, p)):
                if exclude_tracklets is None or t not in exclude_tracklets:  # 这里是来要把查询的轨道进行排除的过程的。
                    pid_list.append(int(p))  # 这里是来添加相应的id号的过程的。
                    tracklets_path.append(os.path.join(home_dir, p + "/" + t)) # 'E:/datasets/DukeMTMC-VideoReID\\train\\0001/0001' 其中第一个0001是对应行人id第二个是对应序列id
                    tracklets_list.append(t) # 这里是来把序列id给添加到里，这里是来把全部的序列id给添加到里面的。
        pid_list = set(pid_list) # 这里是来把相同的id给排除掉的。这里是来得到相应的行人id的过程的。
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}  # 是来对标签进行重置的过程的。这里是打乱后，所对应的过程的。
        tracklets = []  # 这里是来对应的序列号的。
        num_imgs_per_tracklet = [] # 是对应的每个序列，所对应的图片数的过程的。

        # 总共是对应2196个视频序列的。
        for tracklet_idx in range(len(tracklets_path)):
            img_names = os.listdir(tracklets_path[tracklet_idx])  # 这里是来获取里面的图片的
            pid = int(img_names[0].split("_")[0])  # 是来获取id的过程的
            camid = int(img_names[0].split("C")[1].split("_")[0]) # 先分割取后面的部分，然后再来取前面的部分的这样就是能够得到相机号的过程的。

            # 这里是来对属性数据集进行处理的过程的。
            attribute = []
            if attr:
                t_id = int(tracklets_path[tracklet_idx].split("/")[-1]) # 是来按照斜杠进行划分后，取出最后一位的数据的。也就是对应的序列号的

                # 首先是对应的行人id相等 然后是对应的相机id也是相等的，并且是所对应的序列id也是相等的。
                attribute = self.attributes[(self.attributes.person_id == pid) & (self.attributes.camera_id == camid) & (
                        self.attributes.tracklets_id == t_id)].values
                if len(attribute) > 0:
                    attribute = attribute[0, 3:].tolist()  # 是来把前面的一部分数据给去掉，也就是对应的person_id,tracklets_id,cam_id这三个属性给去掉
                    # attribute = attribute[0 : 2] + attribute[9:]
                    # attribute = attribute[2:9].tolist() + attribute[0:2].tolist() + attribute[9:].tolist()
                    # attributes.append(attribute)
                    # attribute = attribute[0, 12:]
            assert 1 <= camid <= 8    # 判断相机号是否是在1-8之间的
            if relabel: pid = pid2label[pid]  # 重新来映射标签的关系的
            camid -= 1  # index starts from 0   # 这里是让相机号是从0开始的。
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]  # 首先是来提取相关的id号的过程的。
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera*
            # 这样也是来确保只有一个相机号，对于每个id的过程的。
            camnames = [img_name[6] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information # 这里是来获取完全的路径信息的。
            img_paths = [osp.join(tracklets_path[tracklet_idx], img_name) for img_name in img_names]

            # 如果是这个id下这个轨道下的全部图片的数量是大于min_seq_len的数量的时候。那么我们的就是要来对图片进行打乱后
            if len(img_paths) >= min_seq_len:
                random.shuffle(img_paths)
                img_paths = tuple(img_paths)  # 获得打乱后，所对应的图片的路径信息的。
                tracklets.append((img_paths, pid, camid, attribute))  # 添加图片路径，行人id,相机id,和所对应的属性的。
                num_imgs_per_tracklet.append(len(img_paths))  # 得到这个轨道序列的长度的，其中所对应的第一序列的长度是177

        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, len(pid_list), num_imgs_per_tracklet, tracklets_list

"""Create dataset"""

__factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid': PRID,
    'duke':DukeMTMC_Video
}

def get_names():
    return __factory.keys()
## 传入数据集的路径和数据集所对应名称的
def init_dataset(name, *args, **kwargs):
    ## 对应的是如果数据集是不在factory字典里面，就是要输出相应的错误的。
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))  # 如果是不在这个字典的里面的，就是要要输出的相应的错误的信息的
    return __factory[name](*args, **kwargs)

if __name__ == '__main__':
    # dataset = iLIDSVID()
    # dataset = PRID()

    # data = Mars(root='D:/datasets/mars')
    data =DukeMTMC_Video(root='E:/datasets/DukeMTMC-VideoReID')







