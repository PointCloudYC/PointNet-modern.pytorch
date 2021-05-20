"""Defines PointNet semantic segmentation model"""
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

import datasets.data_utils as d_utils
from losses import MaskedCrossEntropy

class TNet(nn.Module):
    """align the input or intermediate features, regressing to a kxk matrix
    """
    def __init__(self,k):
        super(TNet, self).__init__()
        self.k=k
        self.conv1 = torch.nn.Conv1d(self.k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.k*self.k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        Args:
            x (tensor): batch point clouds, shape Bx3xN (note: orignal pc is BxNx3)
        Returns:
            y: classification scores
        """
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) # Bx64xN
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) # Bx1024xN
        x,_ = torch.max(x, -1, keepdim=True) # Bx1024x1
        x = x.view(-1, 1024) # Bx1024

        x = F.relu(self.bn4(self.fc1(x))) # Bx512
        x = F.relu(self.bn5(self.fc2(x))) # Bx256
        x = self.fc3(x) #Bxk^2

        # use variable so as to be able to learn them, Bxk^2
        identity = Variable(torch.eye(self.k,dtype=torch.float32).flatten()).view(1,self.k*self.k).repeat(batch_size,1) 
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity # Bxk^2
        x = x.view(-1, self.k,self.k) # Bxkxk
        return x

def conv_bn(in_channels, out_channels, kernel_size=1, batch_norm=True, init_zero_weights=False):
    """Creates a 1d convolutional layer, with optional batch normalization.
    Returns:
        nn.Sequential: the Sequential layers
    """
    layers = []
    conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    return nn.Sequential(*layers)

class PointNetSemSeg(nn.Module):
    def __init__(self, config, input_features_dim):
        """PointNet semantic segmentation model with TNets, whose core components are:
        - MLP: point-wise feature learning for the input cloud
        - max pooling: gain global features for the input cloud
        - FC: 3 fully connected layers to obtain segmentation scores
        - TNet: to achieve input feature transformation invariance

        Args:
            params: (dict) config dict
            input_features_dim: dimension for input feature.

        Returns:
            A series of layers.
        """

        super(PointNetSemSeg, self).__init__()

        self.num_classes = config.num_classes
        self.num_points = config.num_points
        
        self.point_transform=config.point_transform
        if self.point_transform:
            self.tnet1=TNet(3)

        self.conv_bn1=conv_bn(in_channels=3+input_features_dim,out_channels=64,kernel_size=1)
        self.conv_bn2=conv_bn(in_channels=64,out_channels=64,kernel_size=1)

        self.feature_transform=config.feature_transform
        if self.feature_transform:
            self.tnet2=TNet(64)

        self.conv_bn3=conv_bn(in_channels=64,out_channels=64,kernel_size=1)
        self.conv_bn4=conv_bn(in_channels=64,out_channels=128,kernel_size=1)
        self.conv_bn5=conv_bn(in_channels=128,out_channels=1024,kernel_size=1)

        # five MLP(512,256,128,128,m)
        self.conv_bn6=conv_bn(in_channels=1088,out_channels=512,kernel_size=1)
        self.conv_bn7=conv_bn(in_channels=512,out_channels=256,kernel_size=1)
        self.conv_bn8=conv_bn(in_channels=256,out_channels=128,kernel_size=1)
        self.conv_bn9=conv_bn(in_channels=128,out_channels=128,kernel_size=1)
        # Note: BN = False
        self.conv_bn10=conv_bn(in_channels=128,out_channels=self.num_classes,kernel_size=1,batch_norm=False)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, xyz, mask, features):
        """
        Args:
            xyz: (B, N, 3), point coordinates
            mask: (B, N), 0/1 mask to distinguish padding points1
            features: (B, input_features_dim,N), input points features.

        Returns:
            out: (tuple) dimension batch_size x num_classes with the log probabilities for the labels of each image.
        """

        xyz = xyz.permute(0,2,1) # Bx3xN

        # tnet1 for input features
        if self.point_transform:
            transform_input = self.tnet1(xyz)
            xyz = xyz.transpose(2, 1) # BxNx3
            xyz = torch.bmm(xyz, transform_input) # BxNx3
            xyz = xyz.transpose(2, 1) # Bx3xN
        else:
            transform_input = None

        x= torch.cat((xyz,features),dim=1) # Bx(3+input_features_dim)xN

        x = F.relu(self.conv_bn1(x)) # Bx64xN
        x = F.relu(self.conv_bn2(x)) # Bx64xN

        # tnet2 for intermediate point features
        if self.feature_transform:
            transform_feature = self.tnet2(x)
            x = x.transpose(2, 1) # BxNx64
            x = torch.bmm(x, transform_feature) # BxNx64
            local_features= x.transpose(2, 1) # Bx64xN
        else:
            transform_feature = None
            local_features=x # Bx64xN

        x = F.relu(self.conv_bn3(local_features)) # Bx64xN
        x = F.relu(self.conv_bn4(x)) # Bx128xN
        x = F.relu(self.conv_bn5(x)) # Bx1024xN

        # obtain global features using max pooling
        x,_ = torch.max(x,dim=-1) # Bx1024
        
        # concat 
        x = torch.cat((x.unsqueeze(-1).repeat(1,1,self.num_points),local_features), dim=1) # Bx1088xN

        x = F.relu(self.conv_bn6(x)) # Bx512xN
        x = F.relu(self.conv_bn7(x)) # Bx256xN
        x = F.relu(self.conv_bn8(x)) # Bx128xN
        x = F.relu(self.conv_bn9(x)) # Bx128xN
        logits = self.conv_bn10(x) # Bx(num_classes)xN

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        # return F.log_softmax(x, dim=1), transform_point, transform_feature
        return logits, transform_input, transform_feature

def get_masked_CE_loss():
    criterion = MaskedCrossEntropy()
    return criterion

if __name__ == "__main__":
    # obtain config
    import argparse
    from utils.config import config, update_config
    parser = argparse.ArgumentParser('S3DIS semantic segmentation training')
    parser.add_argument('--cfg', type=str, default='project/cfgs/s3dis/pointnet.yaml', help='config file')
    args, unparsed = parser.parse_known_args()
    # update config dict with the yaml file
    update_config(args.cfg)
    print(config)

    # create a model
    model = PointNetSemSeg(config,config.input_features_dim)
    print(model)

    # define a loss
    from losses import MaskedCrossEntropy
    criterion = MaskedCrossEntropy()

    # create a random input and then predict
    batch_size = 2 # config.batch_size  
    num_points = config.num_points
    input_features_dim = config.input_features_dim
    xyz = torch.rand(batch_size,num_points,3)
    mask= torch.ones(batch_size,num_points)
    features = torch.rand(batch_size,input_features_dim,num_points)
    labels = torch.ones(batch_size,num_points,dtype=torch.long)

    preds,_,transform_feature = model(xyz,mask, features)
    print(preds.shape, preds)

    # compute
    loss = criterion(preds,labels,mask,transform_feature)
    print(loss)