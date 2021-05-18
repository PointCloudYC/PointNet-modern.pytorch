#!/bin/bash

# compile custom CUDA operators(sub-sampling and masked related operations)
cd ops/cpp_wrappers
sh compile_wrappers.sh

cd ../pt_custom_ops
python setup.py install --user
cd ../..

# pre-processing all datasets
#python datasets/ModelNet40.py
#python datasets/PartNet.py
#python datasets/S3DIS.py