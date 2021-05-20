# PointNet-modern.pytorch


## Description   
Replicate PointNet using pytorch with good readability and flexibility.

## Preparation

### Requirements
- `Ubuntu 18.04`
- `Anaconda` with `python=3.6`
- `pytorch>=1.5`
- `torchvision` with  `pillow<7`
- `cuda=10.1`
- others: `pip install termcolor opencv-python tensorboard h5py easydict`


### Datasets
**Shape Classification on ModelNet40**

You can download ModelNet40 for [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) (1.6 GB). Unzip and move (or link) it to `data/ModelNet40/modelnet40_normal_resampled`.

**Scene Segmentation on S3DIS**

You can download the S3DIS dataset from [here](https://goo.gl/forms/4SoGp4KtH1jfRqEj2") (4.8 GB). You only need to download the file named `Stanford3dDataset_v1.2.zip`, unzip and move (or link) it to `data/S3DIS/Stanford3dDataset_v1.2`.

The file structure should look like:
```
<project>
├── cfgs
│   ├── modelnet
│   ├── partnet
│   └── s3dis
├── data
│   ├── ModelNet40
│   │   └── modelnet40_normal_resampled
│   │       ├── modelnet10_shape_names.txt
│   │       ├── modelnet10_test.txt
│   │       ├── modelnet10_train.txt
│   │       ├── modelnet40_shape_names.txt
│   │       ├── modelnet40_test.txt
│   │       ├── modelnet40_train.txt
│   │       ├── airplane
│   │       ├── bathtub
│   │       └── ...
│   └── S3DIS
│       └── Stanford3dDataset_v1.2
│           ├── Area_1
│           ├── Area_2
│           ├── Area_3
│           ├── Area_4
│           ├── Area_5
│           └── Area_6
├── init.sh
├── datasets
├── scripts
├── models
├── ops
└── utils
```

### Compile custom operators and pre-processing data
```bash
sh init.sh
```

## How to run   
### Training

#### ModelNet
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node <num_of_gpus_to_use> \
    scripts/train_modelnet.py --cfg <config file> [--log_dir <log directory>]
```
- `<port_num>` is the port number used for distributed training, you can choose like 12347.
- `<config file>` is the yaml file that determines most experiment settings. Most config file are in the `cfgs` directory.
- `<log directory>` is the directory that the log file, checkpoints will be saved, default is `log`.

#### S3DIS
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node <num_of_gpus_to_use> \
    scripts/train_s3dis.py --cfg <config file> [--log_dir <log directory>]
```

### Evaluating
For evaluation, we recommend using 1 gpu for more precise result.
#### ModelNet40
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node 1 \
    scripts/evaluate_modelnet.py --cfg <config file> --load_path <checkpoint> [--log_dir <log directory>]
 ```
- `<port_num>` is the port number used for distributed evaluation, you can choose like 12347.
- `<config file>` is the yaml file that determines most experiment settings. Most config file are in the `cfgs` directory.
- `<checkpoint>` is the model checkpoint used for evaluating.
- `<log directory>` is the directory that the log file, checkpoints will be saved, default is `log_eval`.

#### S3DIS
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node 1 \
    scripts/evaluate_s3dis.py --cfg <config file> --load_path <checkpoint> [--log_dir <log directory>]
```

# Models

## ModelNet40
|Method | Acc | Model |
|:---:|:---:|:---:|
|PointNet|xxx| [Google]() / [Baidu(xxxx)]()|

## S3DIS
|Method | mIoU | Model |
|:---:|:---:|:---:|
|PointNet|xxx| [Google]() / [Baidu(xxxx)]()|


# Python package
## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from project.datasets.S3DIS import S3DISSemSeg
from project.models.pointnet_semseg import PointNetSemSeg
from project.scripts.train_s3dis import train
# from pytorch_lightning import Trainer

# model
model = PointNet(...)

# data
train, test = S3DISSemSeg(...)

# train
loss=train(...)
```

# Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
