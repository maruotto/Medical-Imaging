import torch
import torch.nn as nn
from training.loss import DiceBCELoss
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()

    def forward(self, preds, mask, label, intensity):
        crossEntropy = nn.CrossEntropyLoss()
        binaryCrossEntropy = nn.BCEWithLogitsLoss()  #
        #diceLoss = DiceBCELoss()

        label = label.long()
        intensity = intensity.unsqueeze(1)
        intensity = intensity.float()

        #loss0 = diceLoss._dice_loss(preds[0], mask)
        loss0 = F.binary_cross_entropy_with_logits(preds[0], mask, reduction='mean')
        loss1 = crossEntropy(preds[1], label)
        loss2 = binaryCrossEntropy(preds[2], intensity)

        return torch.stack([loss0, loss1, loss2])
