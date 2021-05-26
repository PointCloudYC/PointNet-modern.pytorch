#!/bin/bash

# compile custom CUDA operators(sub-sampling and related operations)
cd ops/pointnet2_ops
python setup.py install --user
cd ../../../
echo $PWD