import torch
import torch.nn as nn
import torch.nn.functional as F
from training import config
import numpy as np
import torchmetrics.functional as f

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
    
    def _dice_loss(self, inputs, targets):
        targets = targets.type(torch.int64).to(config.DEVICE)
        num_classes = 1
        eye = torch.eye(num_classes + 1).to(config.DEVICE)
        true_1_hot = eye[targets.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(inputs)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, inputs.ndimension()))
        n = torch.mul(probas, true_1_hot).sum(dims)
        n =  torch.mul(2.,n)
        d = torch.add(probas, true_1_hot).sum(dims)
        dice_score = torch.divide(n,d)
        return torch.sub(1, dice_score.mean())

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = torch.add(BCE, self._dice_loss(inputs, targets))
        return Dice_BCE

