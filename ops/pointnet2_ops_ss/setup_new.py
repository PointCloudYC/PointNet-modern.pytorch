# based on https://github.com/sshaoshuai/Pointnet2.PyTorch/blob/master/pointnet2/setup.py
# Modified by Chao YIN

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os.path as osp

_ext_src_dir = "_ext_src"
_ext_files = glob.glob("{}/*.cpp".format(_ext_src_dir)) + glob.glob("{}/src/*.cu".format(_ext_src_dir))
_ext_headers = glob.glob("{}/*.h".format(_ext_src_dir))
current_dir = osp.dirname(osp.abspath(__file__))

setup(
    name='pointnet2_ops',
    ext_modules=[
        CUDAExtension(
            name='pointnet2_ops._ext',
            sources=_ext_files,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_dir))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_dir))],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)