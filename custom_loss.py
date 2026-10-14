import torch
import torch.nn as nn

class CustomCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, label):
        loss = self.ce(logits, label)
        return loss

def build_loss():
    return CustomCrossEntropy()
