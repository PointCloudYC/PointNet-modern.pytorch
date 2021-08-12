"""
from https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2/models/pointnet2_msg_sem.py
modified by Chao YIN 
"""

import torch
import torch.nn as nn

import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models')) # for loss module
sys.path.append(os.path.join(ROOT_DIR, 'ops','pointnet2_ops_lib', 'pointnet2_ops'))
from pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from .pointnet2_ssg_semseg import PointNet2SSGSemSeg


class PointNet2MSGSemSeg(PointNet2SSGSemSeg):
    """
    PointNet2 MSG for semantic segmentation
    """
    def _build_model(self):

        self.SA_modules = nn.ModuleList()
        for i in range(len(self.npoints)):
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=self.npoints[i],
                    radii=self.radii[i], # radii-radii
                    nsamples=self.nsamples[i],
                    mlps=self.mlps[i],
                    use_xyz=self.use_xyz
                )
            )

        self.FP_modules = nn.ModuleList()
        for i in range(len(self.mlps_fp)):
            self.FP_modules.append(PointnetFPModule(mlp=self.mlps_fp[i]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.num_classes, kernel_size=1),
        )
