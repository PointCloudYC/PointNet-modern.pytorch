"""
from https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2/models/pointnet2_ssg_sem.py
modified by Chao YIN 
"""
import torch
import torch.nn as nn

import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'ops','pointnet2_ops_lib', 'pointnet2_ops'))
from pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2_ssg_cls import PointNet2SSGCls


class PointNet2SSGSemSeg(PointNet2SSGCls):
    """
    PointNet2 SSG for semantic segmentation
    """
    def _build_model(self):

        self.SA_modules = nn.ModuleList()
        for i in range(len(self.npoints)):
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self.npoints[i],
                    radius=self.radii[i], # note: radius-radii
                    nsample=self.nsamples[i],
                    mlp=self.mlps[i],
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

    def forward(self, xyz, mask, features):
        """
        Args:
            xyz: (B, N, 3), point coordinates
            mask: (B, N), 0/1 mask to distinguish padding points1
            features: (B, input_features_dim,N), input points features.

        Returns:
            out: (tuple) dimension batch_size x num_classes with the log probabilities for the labels of each pt.
        """

        # xyz = xyz.permute(0,2,1) # Bx3xN

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])

if __name__ == "__main__":

    # obtain config
    import argparse
    from utils.config import config, update_config
    parser = argparse.ArgumentParser('S3DIS semantic segmentation training')
    parser.add_argument('--cfg', type=str, default='project/cfgs/s3dis/pointnet2_ssg.yaml', help='config file')
    args, unparsed = parser.parse_known_args()
    # update config dict with the yaml file
    update_config(args.cfg)
    print(config)

    # create a model
    model = PointNet2SSGSemSeg(config,config.input_features_dim)
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
