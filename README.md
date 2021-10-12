# You Don’t Only Look Once: Constructing Spatial-Temporal Memory for Integrated 3D Object Detection and Tracking
### [Project Page](https://zju3dv.github.io/udolo) | [Paper](https://arxiv.org/pdf/109999.pdf)
<br/>

> You Don’t Only Look Once: Constructing Spatial-Temporal Memory for Integrated 3D Object Detection and Tracking  
> [Jiaming Sun](https://jiamingsun.ml)<sup>\*</sup>, [Yiming Xie](https://ymingxie.github.io)<sup>\*</sup>, [Siyu Zhang](https://derizsy.github.io/), [Linghao Chen](https://f-sky.github.io/), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/), [Hujun Bao](http://www.cad.zju.edu.cn/bao/), [Xiaowei Zhou](http://www.cad.zju.edu.cn/home/xzhou/)  
> ICCV 2021

[comment]: <> (![video]&#40;assets/udolo.gif&#41;)

<br/>

## ETA
The code will released within this week.

## TODO List
- [ ] Code for training and test.
- [ ] Pretrained models on ScanNet.
- [ ] Evaluation code and metrics.

## How to Use

### Installation

Install [Pytorch](https://pytorch.org/get-started/locally/). The code is tested with Python 3.6, Pytorch v1.2.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:
```bash
./install.sh
```
Install the following Python dependencies (with `pip install -r requirements`):


### Pretrained Model on ScanNet
Download the [pretrained weights](https://drive.google.com/file/d/1h3IBhC-oHGmUSEw1zT-AM94gccTqRpFa/view?usp=sharing) and put it under 
`PROJECT_PATH/log_scannet`.
You can also use [gdown](https://github.com/wkentaro/gdown) to download it in command line:
```bash
mkdir log_scannet && cd log_scannet
gdown --id 1h3IBhC-oHGmUSEw1zT-AM94gccTqRpFa
```

### Data Preperation for ScanNet
Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/.  
Download [oriented boxes annotation](https://drive.google.com/file/d/1N-XdIsSpI7PrKgGzwjUk1ZEjykrHr2Z7/view?usp=sharing).
You can also use [gdown](https://github.com/wkentaro/gdown) to download it in command line:
```bash
gdown --id 1N-XdIsSpI7PrKgGzwjUk1ZEjykrHr2Z7
```
<details>
  <summary>[Expected directory structure of ScanNet (click to expand)]</summary>

```
PROJECTROOT
└───scannet
│   └───oriented_boxes_annotation_train.pkl
│   └───oriented_boxes_annotation_val.pkl
│   └───scans
│   |   └───scene0000_00
│   |       └───depth
│   |       │   │   0.png
│   |       │   │   1.png
│   |       │   │   ...
│   |       │   ...
│   └───...
```
</details>


### Train and test on ScanNet

To train a VoteNet model on Scannet data (fused scan):

    python train.py --dataset scannet --log_dir log_scannet --num_point 16384 --batch_size 8

To test the trained model with its checkpoint:

    python eval.py --dataset scannet --checkpoint_path log_scannet/104_checkpoint.tar --dump_dir eval_scannet --num_point 16384 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal --batch_size 1 --faster_eval --integrate

Example results will be dumped in the `eval_scannet` folder (or any other folder you specify). In default we evaluate with both AP@0.25 and AP@0.5 with 3D IoU on oriented boxes.


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{sun2021udolo,
  title={{You Don't Only Look Once}: Constructing Spatial-Temporal Memory for Integrated 3D Object Detection and Tracking},
  author={Sun, Jiaming and Xie, Yiming and Zhang, Siyu and Zhang, Guofeng and Bao, Hujun and Zhou, Xiaowei},
  journal={ICCV},
  year={2021}
}
```

## Acknowledgment
Thanks to Charles R. Qi for opening source of his excellent works [VoteNet](https://github.com/facebookresearch/votenet).
