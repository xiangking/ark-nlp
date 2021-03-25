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

import abc
import dgl
import torch
import random
import numpy as np

from torch.utils.data import Dataset
from .base_tokenizer import BaseTokenizer

    
class TextLevelGCNTokenizer(BaseTokenizer):
    """
    文本编码器，用于对文本进行图编码

    :param max_seq_len: (int) 预设的文本最大长度
    :param tokenizer: (object) 编码器，用于实现文本分词和ID化

    Reference: 
        [1] 
    """  
    def __init__(self, max_seq_len, vocab, graph):
        super(TextLevelGCNTokenizer, self).__init__(max_seq_len, vocab)
        self.graph = graph
        self.tokenizer_type = 'graph'
        
    def text_to_graph(self, text):
        sequence = self.vocab.convert_to_ids(self.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]

        node_ids = list(set(sequence))
        local_token2id = dict(zip(node_ids, range(len(node_ids))))

        sub_graph = dgl.DGLGraph()

        # 节点信息
        sub_graph.add_nodes(len(node_ids))

        # 边和权信息
        local_edges, edge_ids = self.graph.get_sequence_graph(sequence, local_token2id)

        srcs, dsts = zip(*local_edges)
        sub_graph.add_edges(srcs, dsts)

        return node_ids, edge_ids, sub_graph