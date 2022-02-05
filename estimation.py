import torch
import torch.nn as nn

import math


Wq = torch.randn(768, 768)
Wk = torch.randn(768, 768)
Wv = torch.randn(768, 768)

def self_attention(x):
    q = x@Wq
    k = x@Wk
    v = x@Wv
    q = q.view(768, 12, 64).permute(1, 0, 2)
    k = k.view(768, 12, 64).permute(1, 2, 0)
    v = v.view(768, 12, 64).permute(1, 0, 2)

    qk = torch.softmax(q@k, 1) / math.sqrt(64)
    qkv = qk@v
    print(qkv.shape)
    qkv = qkv.permute(1, 0, 2).reshape(768, 768)
    return qkv


class GradientEstimate(nn.Module):

    def __init__(self, n_layer = 2, dim = (768, 768), hidden_state = 64, anchor:torch.Tensor = None) -> None:
        super().__init__()
        self.dim = dim
        self.n_layer = n_layer
        self.hidden_state = hidden_state
        self.F0 = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.F1 = nn.Sequential(
            nn.Linear(dim[0] * 2, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, dim[0])
        )

        self.F2 = nn.Sequential(
            nn.Linear(dim[0] * 2, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, dim[0])
        )
        if anchor is None:
            self.anchor = torch.randn(dim)
        else:
            self.anchor = anchor
    def forward(self, x):
        output = self.F0
        acc = output
        Delta = x - self.anchor
        E = Delta / 1
        next_input = torch.cat([output, E])
        output = self.F1(next_input)
        acc += output

        next_input = torch.cat([output, E])
        output = self.F2(next_input)
        acc += output / 2

        return acc

est = GradientEstimate()
optimizer = torch.optim.SGD(est.parameters(), lr=0.01)

est.cuda()
for i in range(100):
    input_ = torch.randn(768, 768).cuda()
    target = self_attention(input_)
    output = torch.nn.MSELoss(est(input_) - target)
