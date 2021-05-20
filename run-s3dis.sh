#!/bin/bash

# train s3dis
time python
-m torch.distributed.launch --master_port 12346 \
--nproc_per_node 1 \
project/scripts/train_s3dis.py \
--cfg project/cfgs/s3dis/pointnet.yaml
# [--log_dir <log directory>]

# evaluate s3dis
time python
-m torch.distributed.launch --master_port 12346 \
--nproc_per_node 1 \
project/scripts/evaluate_s3dis.py \
--cfg project/cfgs/s3dis/pointnet.yaml
# --load_path <checkpoint>
# [--log_dir <log directory>]
