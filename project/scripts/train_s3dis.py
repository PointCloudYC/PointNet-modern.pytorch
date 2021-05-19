"""
Distributed training script for semantic segmentation on S3DIS dataset
"""
import os
import sys
import time
from datetime import datetime
import json
import random
import numpy as np

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

# configs and logging
import argparse
from utils.config import config, update_config
from utils.logger import setup_logger

# import datasets.data_utils as d_utils
# from models import build_scene_segmentation # models/build.py
# from datasets import S3DISSeg

# metrics and lr scheduler
from utils.util import AverageMeter, s3dis_metrics, sub_s3dis_metrics, s3dis_part_metrics
from utils.lr_scheduler import get_scheduler


def parse_config():
    """load configs including parameters from dataset, model, training, etc.
    The basic process is:
    - load default settings based on the config dict in the utils/config.py
    - update the config dict using yaml file specified by an argparse argument(--cfg argument)
    - update the config dict using argparse arguments

    Returns:
        tuple: (args, config) contains config settings where args is argparse.Namespace object while config is a dict
    """
    parser = argparse.ArgumentParser('S3DIS semantic segmentation training')
    parser.add_argument('--cfg', type=str, default='project/cfgs/s3dis/pointnet.yaml', help='config file')
    parser.add_argument('--data_root', type=str, default='data', help='root director of dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument('--num_steps', type=int, help='num_steps')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight_decay')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, help='used for resume')

    # io
    parser.add_argument('--load_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=10, help='val frequency')
    parser.add_argument('--log_dir', type=str, default='log', help='log dir [default: log]')

    # misc
    parser.add_argument("--local_rank", type=int,default=0, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()

    # update config dict with the yaml file
    update_config(args.cfg)

    # update config dict with args arguments
    config.data_root = args.data_root
    config.num_workers = args.num_workers
    config.load_path = args.load_path
    config.print_freq = args.print_freq
    config.save_freq = args.save_freq
    config.val_freq = args.val_freq
    config.rng_seed = args.rng_seed

    config.local_rank = args.local_rank
    
    model_name = args.cfg.split('.')[-2].split('/')[-1] # model name, e.g., pointnet
    current_time = datetime.now().strftime('%Y%m%d%H%M%S') #20210518221044 means 2021, 5.18, 22:10:44
    config.log_dir = os.path.join(args.log_dir, 's3dis', f'{model_name}_{int(current_time)}') ## log_dir=log/s3dis/pointnet_time 

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_points:
        config.num_points = args.num_points
    if args.num_steps:
        config.num_steps = args.num_steps
    if args.base_learning_rate:
        config.base_learning_rate = args.base_learning_rate
    if args.weight_decay:
        config.weight_decay = args.weight_decay
    if args.epochs:
        config.epochs = args.epochs
    if args.start_epoch:
        config.start_epoch = args.start_epoch

    print(args)
    print(config)

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    return args, config


def main(config):
    pass
    # model

    # loss and optimizer

    # training loop
    # evaluation loop


if __name__ == "__main__":

    # load config
    args, config = parse_config()

    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    os.makedirs(args.log_dir, exist_ok=True)
    os.environ["JOB_LOG_DIR"] = config.log_dir

    logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name="s3dis")
    if dist.get_rank() == 0:
        path = os.path.join(config.log_dir, "config.json")
        # save args and config settings to config.json
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
            json.dump(vars(config), f, indent=2)
            os.system('cp %s %s' % (args.cfg, config.log_dir))
        logger.info("Full config saved to {}".format(path))
    main(config)
