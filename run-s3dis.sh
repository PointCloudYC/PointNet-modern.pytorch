#!/bin/bash

time python -m torch.distributed.launch --master_port 12349 \
--nproc_per_node 1 \
project/scripts/train_s3dis.py \
--cfg project/cfgs/s3dis/pointnet2_ssg.yaml
--load_path log/s3dis/pointnet2_ssg_20210527094215/ckpt_epoch_310.pth
--log_dir log/s3dis/pointnet2_ssg_20210527094215

# time python -m torch.distributed.launch --master_port 12349 \
# --nproc_per_node 1 \
# project/scripts/train_s3dis.py \
# --cfg project/cfgs/s3dis/pointnet2_msg.yaml
# [--log_dir <log directory>]

# time python -m torch.distributed.launch --master_port 12349 \
# --nproc_per_node 1 \
# project/scripts/evaluate_s3dis.py \
# --cfg project/cfgs/s3dis/pointnet.yaml
# --load_path log/s3dis/pointwisemlp_dp_fc1_1617578755/ckpt_epoch_190.pth
# [--log_dir <log directory>]
