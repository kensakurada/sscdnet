import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, mask):
        return self.loss(F.log_softmax(outputs,dim=1), mask)

