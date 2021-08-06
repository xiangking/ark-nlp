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
import warnings
import numpy as np
import pandas as pd

from functools import lru_cache
from torch.utils.data import Dataset
from ark_nlp.dataset.base._dataset import BaseDataset


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target or source[i: i + target_len] == ['##' + target[0]] + target[1:]:
            return i
    return -1


class CasRelREDataset(BaseDataset):

    def __init__(
        self, 
        data, 
        categories=None, 
        is_retain_dataset=False,
        is_train=True,
        is_test=False
    ):

        super(CasRelREDataset, self).__init__(
            data, 
            categories, 
            is_retain_dataset,
            is_train,
            is_test
        )

        warnings.warn("The CasRelREDataset is deprecated, please use other CasRelREDataset ( from ark_nlp.model.re.casrel_bert.casrel_relation_extraction_dataset import CasRelREDataset )", DeprecationWarning)
        
    def _get_categories(self):
        return sorted(list(set([triple_[1] for data_ in self.dataset for triple_ in data_['label']])))
    
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
                triple = (self.tokenizer.tokenize(triple[0]), triple[1], self.tokenizer.tokenize(triple[2]))
                
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])

                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx+1, sub_head_idx + len(triple[0]))
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx+1, obj_head_idx + len(triple[2]), self.cat2id[triple[1]]))

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