import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUR_DIR = os.path.dirname(CUR_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

from datasets.data_utils as d_utils

class MaskedCrossEntropy(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropy, self).__init__()

    def forward(self, logit, target, mask, transform_feature=None):
        """cross entropy with masks

        Args:
            logit (Tensor): logits, shape: Bx(num_classes)xN
            target (Tensor): labels, shape: BxN
            mask (Tensor): mask 0/1, shape: BxN
            transform_feature (Tensor, optional): T-Net feature tranformation matrix(mainly for PointNet). Defaults to None.

        Returns:
            (Tensor): loss, a scalar
        """
        # preserve the shape by setting reduction = none
        loss = F.cross_entropy(logit, target, reduction='none') # BxN
        # omit los of those padded points
        loss *= mask
        loss = loss.sum() / mask.sum()
        
        if transform_feature is not None:
            loss = loss + 0.001 * d_utils.feature_transform_regularizer(transform_feature)
        else:
            loss = loss
        return loss