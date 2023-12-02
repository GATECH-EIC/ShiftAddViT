import torch.nn as nn
import torch

# here we have a or b = mask, MatAdd op can be used to mask out the attention
# but we use multiply here for simplicity
class MatAdd(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        out = a @ b
        return out

# if use Q @ K, FLOPs caclulation could be wrong
class MatMul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        out = a @ b
        return out


class Mul(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        out = a * b
        return out

