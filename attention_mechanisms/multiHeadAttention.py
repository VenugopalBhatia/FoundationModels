from selfAttention import SelfAttention
import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self,d,dq,dk,dv,numHeads):
        super().__init__()
        self.attnHeads = nn.ModuleList(
            [SelfAttention(d,dq,dk,dv) for _ in range(numHeads)]
        )
    def forward(self,x):
        return torch.cat([head(x) for head in self.attnHeads],dim = -1)
    

