"""
# Copyright Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
From: https://github.com/suolyer/PyTorch_BERT_Biaffine_NER
"""

import torch
from torch import nn


class Biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(
            torch.randn(in_size + int(bias_x), out_size, in_size + int(bias_y))
        )

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        # batch_size,seq_len,hidden=x.shape
        # bilinar_mapping=torch.matmul(x,self.U)
        # bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        # y=torch.transpose(y,dim0=1,dim1=2)
        # bilinar_mapping=torch.matmul(bilinar_mapping,y)
        # bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        # bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)

        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping
