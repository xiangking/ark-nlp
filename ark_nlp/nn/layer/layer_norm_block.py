import torch
import torch.nn as nn

from typing import Optional


class CondLayerNormLayer(nn.Module):
    def __init__(self, hidden_size: int, eps: Optional[float] = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.variance_epsilon = eps
        self.weight_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.weight_linear.weight.data)
        self.bias_linner = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.bias_linner.weight.data)

    def forward(self, x, cond):
        for _ in range(x.dim() - cond.dim()):
            cond = cond.unsqueeze(1)
        weight = self.weight + self.weight_linear(cond)
        bias = self.bias + self.bias_linner(cond)
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return weight * x + bias
