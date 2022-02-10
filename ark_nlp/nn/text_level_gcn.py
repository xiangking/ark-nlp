# Copyright (c) 2021 DataArk Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xiang Wang, xiangking1995@163.com
# Status: Active


import dgl
import torch

from torch import nn


class TextLevelGCN(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        node_embed_size,
        edge_num,
        class_num=2,
        pre_node_embed=False,
        node_embed_vectors=0,
        is_node_freeze=False,
        pre_edge_embed=False,
        edge_embed_vectors=0,
        is_edge_freeze=False,
        fc_dropout_rate=0.5
    ):
        super(TextLevelGCN, self).__init__()

        self.class_num = class_num

        self.vocab_size = vocab_size
        self.node_embed_size = node_embed_size

        self.pre_node_embed = pre_node_embed
        self.node_freeze = is_node_freeze

        if self.pre_node_embed:
            self.node_embed = nn.Embedding.from_pretrained(
                embeddings=node_embed_vectors,
                freeze=self.node_freeze
            )
        else:
            self.node_embed = nn.Embedding(
                self.vocab_size,
                self.node_embed_size
            )
            nn.init.uniform_(self.node_embed.weight.data)

        self.pre_edge_embed = pre_edge_embed
        self.edge_num = edge_num
        self.edge_freeze = is_edge_freeze

        if self.pre_edge_embed:
            self.edge_embed = torch.nn.Embedding.from_pretrained(
                embeddings=edge_embed_vectors,
                freeze=self.edge_freeze
            )
        else:
            self.edge_embed = torch.nn.Embedding.from_pretrained(
                torch.ones(edge_num, 1),
                freeze=self.edge_freeze
            )

        self.dropout = torch.nn.Dropout(p=fc_dropout_rate)

        self.activation = torch.nn.ReLU()

        self.classify = torch.nn.Linear(self.node_embed_size, self.class_num, bias=True)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.classify.weight)

    def forward(
        self,
        sub_graph,
        **kwargs
    ):
        sub_graph.update_all(
            message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
            reduce_func=dgl.function.max('weighted_message', 'h')
        )
        graph_feature = dgl.sum_nodes(sub_graph, feat='h')

        graph_feature = self.dropout(graph_feature)
        graph_feature = self.activation(graph_feature)

        out = self.classify(graph_feature)

        return out
