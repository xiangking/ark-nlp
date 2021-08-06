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

import copy
import json
import torch
import random
import codecs
import numpy as np
import pandas as pd

from functools import lru_cache
from torch.utils.data import Dataset
from ark_nlp.dataset.base._dataset import BaseDataset


class CasRelREDataset(BaseDataset):
        
    def _get_categories(self):
        return sorted(list(set([triple_[3] for data_ in self.dataset for triple_ in data_['label']])))
    
    def _convert_to_dataset(self, data_df):
        
        dataset = []
        
        data_df['text'] = data_df['text'].apply(lambda x: x.strip())
        if not self.is_test:
            data_df['label'] = data_df['label'].apply(lambda x: eval(x))
                        
        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
 
            dataset.append({feature_name_: getattr(row_, feature_name_) 
                             for feature_name_ in feature_names})
        return dataset
    
    def convert_to_ids(self, tokenizer):
        """
        将文本转化成id的形式
        
        :param tokenizer:
        
        ToDo: 将__getitem__部分ID化代码迁移到这部分
        
        """
        self.tokenizer = tokenizer
            
        if self.is_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)
                    
    def __getitem__(self, idx):
        ins_json_data = self.dataset[idx]
        text = ins_json_data['text']
        
        if len(text) > self.tokenizer.max_seq_len - 2:
            text = text[:self.tokenizer.max_seq_len - 2]
        
        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens, is_mapping_index=False)
        index_token_mapping = self.tokenizer.get_token_mapping(text, tokens)
        
        start_mapping = {j[0]: i for i, j in enumerate(index_token_mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(index_token_mapping) if j}

        if not self.is_train:
            token_ids, masks, segment_ids = self.tokenizer.sequence_to_ids(text)
            text_len = len(token_ids)
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
            obj_heads, obj_tails = np.zeros((text_len, self.class_num)), np.zeros((text_len, self.class_num))
            if self.is_test:
                return token_ids, masks, text_len, tokens
            else:
                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, ins_json_data['label'], tokens, token_mapping
        else:
            s2ro_map = {}
            for triple in ins_json_data['label']:
                sub_head_idx = triple[1]
                sub_end_idx = triple[2]
                obj_head_idx = triple[5]
                obj_end_idx = triple[6]
                
                triple = (self.tokenizer.tokenize(triple[0]), triple[3], self.tokenizer.tokenize(triple[4]))
                
                if sub_head_idx in start_mapping and obj_head_idx in start_mapping and sub_end_idx in end_mapping and obj_end_idx in end_mapping:
                    sub_head_idx = start_mapping[sub_head_idx]
                    obj_head_idx = start_mapping[obj_head_idx]
                                        
                    sub = (sub_head_idx+1, end_mapping[sub_end_idx]+1)
                    
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    
                    s2ro_map[sub].append((obj_head_idx+1, end_mapping[obj_end_idx]+1, self.cat2id[triple[1]]))

            if s2ro_map:
                token_ids, masks, segment_ids = self.tokenizer.sequence_to_ids(text)
                text_len = len(token_ids)
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                sub_head_idx, sub_tail_idx = random.choice(list(s2ro_map.keys()))
                sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                obj_heads, obj_tails = np.zeros((text_len, self.class_num)), np.zeros((text_len, self.class_num))
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1
                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, ins_json_data['label'], tokens, token_mapping
            else:
                return None