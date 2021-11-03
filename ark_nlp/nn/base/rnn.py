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

from ark_nlp.nn import BasicModule
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


class RNN(BasicModule):
    """
    RNN系列模型封装，包括原始的RNN，LSTM和GRU

    Args:

    Reference: 
        [1] https://github.com/aehrc/LAAT  
    """  # noqa: ignore flake8"

    def __init__(
        self,
        vocab_size,
        embed_size,
        class_num=2,
        embed_dropout=0.2,
        pre_embed=False,
        emb_vectors=0,
        is_freeze=False,
        hidden_size=100,
        hidden_num=1,
        lstm_dropout=0,
        is_bidirectional=True,
        fc_dropout=0.5,
        rnn_cell='lstm'
    ):
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.class_num = class_num

        self.embed_dropout = embed_dropout
        self.pre_embed = pre_embed
        self.freeze = is_freeze

        # 是否加载预训练的词向量,（ batchsize,所截取的序列长度(记为step),词向量维度(embed_size) ）
        if self.pre_embed is True:
            self.embed = nn.Embedding.from_pretrained(
                embeddings=emb_vectors,
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

        self.hidden_size = hidden_size
        self.hidden_num = hidden_num
        self.hidden_dropout = lstm_dropout
        self.bidirectional = is_bidirectional

        self.rnn_cell = rnn_cell

        if self.rnn_cell == 'rnn':
            self.rnn = nn.RNN(
                self.embed_size,
                self.hidden_size,
                num_layers=self.hidden_num,
                dropout=self.hidden_dropout,
                bidirectional=self.bidirectional
            )
        elif self.rnn_cell == 'lstm':
            self.rnn = nn.LSTM(
                self.embed_size,
                self.hidden_size,
                dropout=self.hidden_dropout,
                num_layers=self.hidden_num,
                bidirectional=self.bidirectional
            )
        elif self.rnn_cell == 'gru':
            self.rnn = nn.GRU(
                self.embed_size,
                self.hidden_size,
                dropout=self.hidden_dropout,
                num_layers=self.hidden_num,
                bidirectional=self.bidirectional
            )
        else:
            raise ValueError('错误！没有这种RNN模型')

        if self.bidirectional:
            self.lstm_out_dim = self.hidden_size * 2
        else:
            self.lstm_out_dim = self.hidden_size

        self.fc_dropout = fc_dropout

        self.linear = nn.Linear(self.lstm_out_dim, self.lstm_out_dim // 2)
        self.classify = nn.Linear(self.lstm_out_dim // 2, self.class_num)

        self.fc_dropout = nn.Dropout(self.fc_dropout)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.classify.weight)

    def init_hidden(self, batch_size, num_layers, device):
        if self.rnn_cell == 'lstm':
            return (Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).to(device),
                    Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).to(device))
        else:
            return (Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).to(device))

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output

    def forward(
        self,
        input_ids: torch.LongTensor,
        length: torch.LongTensor,
        **kwargs
    ):
        device = input_ids.device
        batch_size = input_ids.size()[0]

        out = self.embed(input_ids)
        out = self.embed_dropout(out)

        if self.bidirectional is True:
            hidden = self.init_hidden(batch_size, 2, device)
        else:
            hidden = self.init_hidden(batch_size, 1, device)

        self.rnn.flatten_parameters()
        out = pack_padded_sequence(
            out,
            length,
            batch_first=True,
            enforce_sorted=False
        )
        _, hidden = self.rnn(out, hidden)
        if self.rnn_cell == 'lstm':
            hidden = hidden[0]

        hidden = self.get_last_hidden_output(hidden)

        out = self.linear(F.relu(hidden))
        out = self.fc_dropout(out)

        output = self.classify(F.relu(out))

        return output
