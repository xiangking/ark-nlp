"""
# Copyright 2020 Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

import torch
import torch.nn.functional as F

from torch import nn


class TextCNN(torch.nn.Module):
    """
    TextCNN模型封装
    """
    def __init__(
        self,
        vocab_size,
        embed_size,
        class_num=2,
        embed_dropout=0.2,
        pre_embed=False,
        embed_vectors=0,
        is_freeze=False,
        kernel_num=100,
        kernel_size=[3, 5, 7],
        stride=1,
        fc_dropout_rate=0.5
    ):
        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embed_dropout = embed_dropout
        self.pre_embed = pre_embed
        self.freeze = is_freeze

        if self.pre_embed is True:
            self.embed = nn.Embedding.from_pretrained(
                embeddings=embed_vectors,
                freeze=self.freeze
            )
            nn.init.normal_(self.embed.weight.data[0])
            nn.init.normal_(self.embed.weight.data[1])
        else:
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            nn.init.uniform_(self.embed.weight.data)
            nn.init.normal_(self.embed.weight.data[0])
            nn.init.normal_(self.embed.weight.data[1])

        self.embed_dropout = nn.Dropout(self.embed_dropout)

        self.stride = stride
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        self.convs = nn.ModuleList(
            [nn.Conv2d(
                1,
                self.kernel_num,
                (K, self.embed_size),
                stride=self.stride, padding=(K // 2, 0))
                for K in self.kernel_size]
        )

        covfeature_num = len(self.kernel_size) * self.kernel_num
        self.fc_dropout_rate = fc_dropout_rate
        self.class_num = class_num

        self.linear = nn.Linear(covfeature_num, covfeature_num // 2)
        self.classify = nn.Linear(covfeature_num // 2, self.class_num)
        self.fc_dropout = nn.Dropout(self.fc_dropout_rate)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.classify.weight)

    def forward(self, input_ids, **kwargs):

        out = self.embed(input_ids)
        out = self.embed_dropout(out)

        cov_list = []
        out = out.unsqueeze(1)
        for conv in self.convs:
            cov_list.append(torch.tanh(conv(out)).squeeze(3))
        out = cov_list

        pool_list = []
        for conv_Tensor in out:
            pool_list.append(
                F.max_pool1d(
                    conv_Tensor,
                    kernel_size=conv_Tensor.size(2)).squeeze(2)
            )

        out = torch.cat(pool_list, 1)

        out = self.linear(F.relu(out))
        out = self.fc_dropout(out)
        out = self.classify(F.relu(out))

        return out
