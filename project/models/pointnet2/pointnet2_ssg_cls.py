"""
from https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2/models/pointnet2_ssg_cls.py
modified by Chao YIN 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models')) # for loss module
sys.path.append(os.path.join(ROOT_DIR, 'ops','pointnet2_ops_lib', 'pointnet2_ops'))
from pointnet2_modules import PointnetSAModule


class PointNet2SSGCls(nn.Module):
    def __init__(self, config, input_features_dim):
        super().__init__()

        """PointNet2 SSG for classification
        Args:
            params: (dict) config dict
            input_features_dim: dimension for input feature.
        """
        self.config = config
        self.input_features_dim = input_features_dim

        # load set abtraction config
        self.npoints=config.npoints
        self.radii=config.radii
        self.nsamples=config.nsamples
        self.mlps=config.mlps
        self.mlps_fp=config.mlps_fp # note: for cls, it is []
        self.use_xyz=config.use_xyz
        self.num_classes = config.num_classes
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()

        for i in range(len(self.radii)):
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self.npoints[i],
                    radius=self.radii[i],
                    nsample=self.nsamples[i],
                    mlp=self.mlps[i],
                    use_xyz=self.use_xyz
                )
            )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=self.mlps[-1],
                use_xyz=self.use_xyz
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256,self.num_classes)
        )

    def forward(self, xyz, mask, features):
        """
        Args:
            xyz: (B, N, 3), point coordinates
            mask: (B, N), 0/1 mask to distinguish padding points1
            features: (B, input_features_dim,N), input points features.

        Returns:
            out: (tuple) dimension batch_size x num_classes with the log probabilities for the labels of each pt.
        """

        xyz = xyz.permute(0,2,1) # Bx3xN

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

if __name__ == "__main__":
    # obtain config
    import argparse
    from utils.config import config, update_config
    parser = argparse.ArgumentParser('ModelNet40 classification training')
    parser.add_argument('--cfg', type=str, default='project/cfgs/modelnet/pointnet2_ssg.yaml', help='config file')
    args, unparsed = parser.parse_known_args()
    # update config dict with the yaml file
    update_config(args.cfg)
    print(config)

    # create a model
    model = PointNet2SSGCls(config,config.input_features_dim)
    print(model)
    # IMPORTANT: place model to GPU so that be able to test GPU CUDA ops
    if torch.cuda.is_available():
        model=model.cuda()

    # define a loss
    from losses import MaskedCrossEntropy
    criterion = MaskedCrossEntropy()

    # create a random input and then predict
    batch_size = 2 # config.batch_size  
    num_points = config.num_points
    input_features_dim = config.input_features_dim
    # IMPORTANT: place these tensors to GPU so that be able to test GPU CUDA ops
    xyz = torch.rand(batch_size,num_points,3).cuda()
    mask= torch.ones(batch_size,num_points).cuda()
    features = torch.rand(batch_size,input_features_dim,num_points).cuda()
    labels = torch.ones(batch_size,num_points,dtype=torch.long).cuda()

    # torch.cuda.set_device(0)
    preds = model(xyz, mask, features)
    print(preds.shape, preds)

    # compute
    loss = criterion(preds,labels,mask)
    print(loss)
