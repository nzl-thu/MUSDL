MUSDL
===
This repository contains the PyTorch implementation for paper __Uncertainty-aware Score Distribution Learning for Action Quality Assessment__ (CVPR 2020) \[[arXiv](https://arxiv.org/abs/2006.07665)\]

<img src="https://github.com/nzl-thu/MUSDL/blob/master/fig/multi-path.png" width="60%" alt=""/>

If you find our work useful in your research, please consider citing:
```
@inproceedings{musdl,
  title={Uncertainty-Aware Score Distribution Learning for Action Quality Assessment},
  author={Tang, Yansong and Ni, Zanlin and Zhou, Jiahuan and Zhang, Danyang and Lu, Jiwen and Wu, Ying and Zhou, Jie},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
## Usage

### Requirement

   
- Python >= 3.6
- Pytorch >= 0.4.0


### Dataset Preparation
#### MTL-AQA dataset

MTL-AQA dataset was orignially presented in the paper __What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment__ (CVPR 2019) \[[arXiv](https://arxiv.org/abs/1904.04346)\], where the authors provided the YouTube links of untrimmed long videos and the corresponding annotations at [here](https://github.com/ParitoshParmar/MTL-AQA/tree/master/MTL-AQA_dataset_release).

You can download our prepared MTL-AQA dataset (About 1 G) from [Google Drive](https://drive.google.com/open?id=1T7bVrqdElRLoR3l6TxddFQNPAUIgAJL7). Then, please move the uncompressed data folder to `MTL-AQA/data/frames`. We used the I3D backbone pretrained on Kinetics([Google Drive](https://drive.google.com/open?id=1M_4hN-beZpa-eiYCvIE7hsORjF18LEYU)), which is referenced from [Gated-Spatio-Temporal-Energy-Graph](https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph). The whole data structure should be:
```
./MTL-AQA
├── data
|  └── frames
|  └── info
|  └── rgb_i3d_pretrained.pt
├── dataset.py
├── models.py
...
```

#### JIGSAWS dataset
JIGSAWS dataset was presented in the paper __Jhu-isi gesture and skill assessment working set (jigsaws): A surgical activity dataset for human motion modeling__ (MICCAI workshop 2014), where the raw videos could be downloaded at [here](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/).
You can download our prepared JIGSAWS frames (About 500 M) at [Google Drive](https://drive.google.com/open?id=1VJZHuD7uXeDjYAPNsoKJdUXXSr5V4LSS).

When you have prepared the data, the structure should be:
```
./JIGSAWS
├── data
|  └── frames
|  └── info
|  └── rgb_i3d_pretrained.pt
├── dataset.py
├── models.py
...
```

These data could be also downloaded from [BaiduYun](https://github.com/nzl-thu/MUSDL/blob/master/Baidu-link.txt)

#### AQA-7 dataset
There's an ongoing project regarding this dataset and the code will be public later, please email us if you're interested.

### Training & Evaluation
To train and evaluate the USDL&MUSDL model on MTL-AQA:
```
cd ./MTL-AQA
bash run.sh
```
Note: Recommend using more than 1 GPU for training. You may want to change the gpu ids in run.sh.

To train and evaluate the USDL&MUSDL model on JIGSAWS:
```
cd ./JIGSAWS
bash run.sh
```

Note: Recommend using more than 1 GPU for training. You may want to change the gpu ids in run.sh.

To obatain the average performance across multiple actions by the [file](https://github.com/nzl-thu/MUSDL/blob/master/JIGSAWS/JIGSAWS_avg.xls).

## Acknowledgement
The authors would sincerely thank Xumin Yu from THU (one of the contributors of this repository) for conducting partial experiments in this paper, Paritosh Parmar from UNLV for sharing his codes, Wanhua Li from THU for valuable discussion on the USDL approach, and Jia-Hui Pan from SYSU for re-implementing the baseline model.

## Contact
If you have any questions about our work, please contact <tys15@mails.tsinghua.edu.cn>, <nzl17@mails.tsinghua.edu.cn>
