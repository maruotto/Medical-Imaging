import torch
import torch.nn as nn
from training.loss import DiceBCELoss

class MultiTaskloss(nn.Module):
    def __init__(self):
        super(MultiTaskloss, self).__init()

    def forward(self, preds, mask, label, intensity):
        crossEntropy = nn.CrossEntropyLoss()
        binaryCrossEntropy = nn.BCEWithLogitsLoss()
        diceLoss = DiceBCELoss()
        label = label.long()
        intensity = intensity.unsqueeze(1)
        intensity = intensity.float()
        loss0 = diceLoss._dice_loss(preds[0], mask)
        loss1 = crossEntropy(preds[1], label)
        loss2 = binaryCrossEntropy(preds[2], intensity)

        return torch.stack([loss0, loss1, loss2])
