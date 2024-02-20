import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self,d,dq,dk,dv):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.Wq = torch.nn.Parameter(torch.rand(d,dq))
        self.Wk = torch.nn.Parameter(torch.rand(d,dk))
        self.Wv = torch.nn.Parameter(torch.rand(d,dv))
    def forward(self,x):
        # x = embedded_sentence
        q = x@self.Wq
        k = x@self.Wk
        v = x@self.Wv

        # omega = Unnormalised attention
        omega = q@k.T

        attention_weights = torch.softmax(
            omega/self.dk**0.5
            ,dim = -1
            )
        context_vector =  attention_weights@v
        # (n X dv) where n = num_words
        return context_vector


