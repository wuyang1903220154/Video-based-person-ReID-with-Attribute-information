# Video-based-person-ReID-with-Attribute-information

This is the code repository for our paper "Attribute-aware Identity-hard Triplet Loss for Video-based Person Re-identification": https://arxiv.org/pdf/2006.07597.pdf.
If you find this help your research, please cite it.
```
@article{chen2020attribute,
  title={Attribute-aware Identity-hard Triplet Loss for Video-based Person Re-identification},
  author={Chen, Zhiyuan and Li, Annan and Jiang, Shilu and Wang, Yunhong},
  journal={arXiv preprint arXiv:2006.07597},
  year={2020}
}
```
### 介绍
##### 对于这篇论文，主要是运用了类内的属性损失，目的是来减少，相同id视频序列内的方差，并且是使用了联合属性框架，来提高识别的准确率。其中我对这篇论文代码有了全面的注释，几乎是每一行代码都被注释了，非常适合学习，如果你想测试这个代码的时候，我们可以使用 from IPython import embed这个包来测试代码，这样 在你要测试代码前面加入embed()这个函数，重新来运行代码，然后你就是可以查看你输出的效果的。相当于设置断点，由于我们实验室设备的限制，我的结果还是不达到与原来论文相同的结果的。但结果也是非常接近的。但是我在改小batch和number_worker的一些参数之后，准确率会下降，如果在我这样的实验条件下，在增大batch和number_worker准确率会上升，所以我有理由相信，当我的实验条件达到论文中的实验条件，会达到原论文的准确率。
### Introduction
This repository contains a project which firstly introducing the pedestrain attribute information into video-based Re-ID, we address this issue by introducing a new metric learning method called Attribute-aware Identity-hard Triplet Loss (AITL), which reduces the intra-class variation among positive samples via calculating attribute distance. To achieve a complete model of video-based person Re-ID, a multitask framework with Attribute-driven Spatio-Temporal Attention (ASTA) mechanism is also proposed. 
#### 1. Attribute-aware Identity-hard Triplet Loss 
The batch-hard triplet loss frequently used in video-based person Re-ID suffers from the Distance Variance among Different Positives(DVDP) problem.
![DVDP](./display_images/pic.png)

Attribute-aware Identity-hard Triplet Loss to solve the DVDP.
![AITL](./display_images/pic2.png)

#### 2. Attribute-driven Spatio-Temporal Attention 
Introducing the spatial-temporal attention in attribute recognition process into Re-ID process.
![ASTA](./display_images/pic1.png)

### Deployment
It is mainly forked from [video-person-reid](https://github.com/jiyanggao/Video-Person-ReID) and [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline). Since I suffered from severe poverty, I introduce the [nvidia-apex](https://github.com/NVIDIA/apex) to train the model in FP16 settings, so the training codes can be directly ran on a single RTX2070s, which is very friendly to proletarians like me. 
If you owes a 32GB V100 Graphic Card or 2 * GTX 1080Ti Cards, you can just ignore the apex operation and run the codes on a single card, and increase the batch size to 64, the u can get a higher performance :).

Requirements:
```
pytorch >= 0.4.1 ( < 1.5.0 apex is not friendly to pytorch 1.5.0 according to my practice)
torchvision >= 0.2.1
tqdm
[nvidia-apex](https://github.com/NVIDIA/apex), please follow the detailed install instructions 
```


### Dataset
#### MARS
Experiments on MARS, as it is the largest dataset available to date for video-based person reID. Please follow [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) to prepare the data. The instructions are copied here: 

1. Create a directory named `mars/`.
2. Download dataset to `mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). The data structure would look like:
5. Download `mars_attributes.csv` from http://irip.buaa.edu.cn/mars_duke_attributes/index.html, and put the file in `data/mars`. The data structure would look like:
```
mars/
    bbox_test/
    bbox_train/
    info/
    mars_attributes.csv
```
6. Change the global variable `_C.DATASETS.ROOT_DIR` to `/path2mars/mars` and `_C.DATASETS.NAME` to `mars` in config or configs.

#### Duke-VID
1. Create a directory named `duke/` under `data/`.
2. Download dataset to `data/duke/` from http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip.
3. Extract `DukeMTMC-VideoReID.zip`.
4. Download `duke_attributes.csv` from http://irip.buaa.edu.cn/mars_duke_attributes/index.html, and put the file in `data/duke`. The data structure would look like:
```
duke/
    train/
    gallery/
    query/
    duke_attributes.csv
```
5. Change the global variable `_C.DATASETS.ROOT_DIR` to `/path2duke/duke` and `_C.DATASETS.NAME` to `duke` in config or configs.

### Usage
To train the model, please run

    python main_baseline.py
 
Please modifies the settings directly on the config files.   


### Performance


#### Comparision with SOTA
![Comparision with SOTA](./display_images/pic4.png)
***The above performance is achieved in the setting: 2 * 1080Ti, train batchsize 64. (Once i was a middle-class deepnetwork-finetuner when i was in school.)***

**Best performance on lower devices(MARS, 1 * RTX 2070s, train batchsize 32)**: (Now i'm a proletarian. 要为了真理而斗争！)

mAP : 82.5%  Rank-1 : 86.5%

#### Better trade-off between speed and performance:
![Computation-performance Balance](./display_images/pic3.png)

More experiments result can be found in paper.
