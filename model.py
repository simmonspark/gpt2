import math

import torch
import torch.nn as nn
from torch.nn import functional as F



class NewGELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super(CausalSelfAttention, self).__init__()
