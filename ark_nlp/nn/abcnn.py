"""
# Copyright Xiang Wang, Inc. All Rights Reserved
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
from ark_nlp.nn import BasicModule


class ABCNN(torch.nn.Module):
    """
    封装了TextCNN模型
    """  
    def __init__(
        self, 
        dic_size,
        emb_dim, 
        class_num=2, 
        embed_dropout=0.2, 
        pre_embed=False, 
        emb_vectors=0, 
        is_freeze=False,
        kernel_num=100, 
        kernel_size=[3, 5, 7], 
        stride=1, 
        fc_dropout_rate=0.5,
        num_layer=1,
        max_seq_len=50,
        linear_size=
    ):
        super(ABCNN, self).__init__()

        self.word_num = dic_size
        self.embed_dim = emb_dim

        self.embed_dropout = embed_dropout
        self.pre_embed = pre_embed
        self.freeze = is_freeze

        if self.pre_embed is True:
            self.embed = nn.Embedding.from_pretrained(embeddings=emb_vectors, freeze=self.freeze)
            nn.init.normal_(self.embed.weight.data[0])
            nn.init.normal_(self.embed.weight.data[1])
        else:
            self.embed = nn.Embedding(len_dic, emb_dim)
            nn.init.uniform_(self.embed.weight.data)
            nn.init.normal_(self.embed.weight.data[0])
            nn.init.normal_(self.embed.weight.data[1])

        self.embed_dropout = nn.Dropout(self.embed_dropout)

        self.conv = nn.ModuleList([Wide_Conv(kernel_size, embeddings.shape[1], device) for _ in range(self.num_layer)])

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
            pool_list.append(F.max_pool1d(conv_Tensor,
                                          kernel_size=conv_Tensor.size(2))
                             .squeeze(2)) 
        out = torch.cat(pool_list, 1)

        out = self.linear(F.relu(out)) 
        out = self.fc_dropout(out)
        out = self.classify(F.relu(out)) 

        return out