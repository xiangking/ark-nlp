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

try:
    import dgl
except:
    ImportWarning('DGL Not Found')
import copy
import torch
import pandas as pd

from functools import lru_cache
from torch.utils.data import Dataset
from .tc_dataset import TCDataset


class TextLevelGCNDataset(TCDataset):
    def __init__(
        self,
        data_path, 
        categories=None, 
        is_retain_dataset=False,
    ):
        super(TCDataset, self).__init__(data_path, categories, is_retain_dataset)
        
    def convert_to_ids(self, tokenizer):
        """
        将文本转化成id的形式
        
        :param tokenizer: 
        """         
        if tokenizer.tokenizer_type == 'graph':
            features = self._convert_to_graph_ids(tokenizer)
        else:
            raise ValueError("The tokenizer type does not exist") 
            
        if self.is_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)
            
        self.dataset = features
        
    def _convert_to_graph_ids(self, graph_tokenizer):
        
        features = []
        for (index_, row_) in enumerate(self.dataset):
            node_ids, edge_ids, sub_graph = graph_tokenizer.text_to_graph(row_['text'])             
                        
            label_ids = self.cat2id[row_['label']]
                        
            features.append({
                'node_ids': node_ids,
                'edge_ids': edge_ids,
                'sub_graph': sub_graph,
                'label_ids': label_ids
            })
            
        return features   
