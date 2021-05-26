"""
no longer needed since pointnet2_ssg_cls can provide this form 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops','pointnet2_ops_lib', 'pointnet2_ops'))
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
# from .pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from .pointnet2_ssg_cls import PointNet2SSGCls


class PointNet2MSGCls(PointNet2SSGCls):
    """PointNet2 MSG for classification
    """
    def _build_model(self):

        # call the base method and then override SA_modules
        super()._build_model()

        self.SA_modules = nn.ModuleList()

        for i in range(len(self.radii)):
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=self.npoints[i],
                    radii=self.radii[i],
                    nsamples=self.nsamples[i],
                    mlps=self.mlps[i],
                    use_xyz=self.use_xyz
                )
            )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=self.mlps[-1],
                use_xyz=self.use_xyz
            )
        )
