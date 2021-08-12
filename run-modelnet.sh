#!/bin/bash

# train modelnet
time python -m torch.distributed.launch --master_port 12346 \
--nproc_per_node 1 \
project/scripts/train_modelnet.py \
--cfg project/cfgs/modelnet/pointnet.yaml
# [--log_dir <log directory>]

# evaluate modelnet
time python -m torch.distributed.launch --master_port 12346 \
--nproc_per_node 1 \
project/scripts/evaluate_modelnet.py \
--cfg project/cfgs/modelnet/pointnet.yaml
# --load_path log/modelnet/pointwisemlp_dp_fc1_1617578755/ckpt_epoch_190.pth
# [--log_dir <log directory>]