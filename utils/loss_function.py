import torch
import torch.nn as nn
import torch.nn.functional as F


class multi_classification_loss(nn.Module):
    def __init__(self):
        super(multi_classification_loss, self).__init__()

    def forward(self,logits,label):
        # loss = - label * F.logsigmoid(logits) - (1-label) * F.logsigmoid(1-logits)
        loss = - label * F.logsigmoid(logits) - (1 - label) * F.logsigmoid(- logits)
        loss = torch.mean(loss)

        return loss
