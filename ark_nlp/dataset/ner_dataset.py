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
import torch
import pandas as pd

from functools import lru_cache
from torch.utils.data import Dataset
from .base_dataset import BaseDataset

import os
import re
from functools import lru_cache

import json
import jieba
import codecs
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from zhon.hanzi import punctuation


class NERDataset(BaseDataset):
    def __init__(
        self,
        data_path, 
        categories=None, 
        is_retain_dataset=False
    ):
        super(NERDataset, self).__init__(data_path, categories, is_retain_dataset)
        
    def _get_categories(self):
        categories = sorted(list(set([label_ for data in self.dataset for label_ in data['label']])))
        categories.remove('O')
        categories.insert(0, 'O')        
        return categories
    
    def _convert_to_dataset(self, data_df):
        
        dataset = []
        
        data_df['text'] = data_df['text'].apply(lambda x: x.lower().strip())
        data_df['label'] = data_df['label'].apply(lambda x: eval(x))
        
        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({feature_name_: getattr(row_, feature_name_) 
                             for feature_name_ in feature_names})
            
        return dataset

    def _convert_to_transfomer_ids(self, bert_tokenizer):
        
        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = bert_tokenizer.text_to_sequence(row_['text'])              
            
            input_ids, input_mask, segment_ids = input_ids
            
            label_ids = bert_tokenizer.pad_and_truncate([self.cat2id[label_] for label_ in row_['label']], bert_tokenizer.max_seq_len - 2)
            label_ids = np.array([self.cat2id['O']] + list(label_ids) + [self.cat2id['O']])
            
            input_length = self._get_input_length(row_['text'], bert_tokenizer)
            
            features.append({
                'input_ids': input_ids, 
                'attention_mask': input_mask, 
                'token_type_ids': segment_ids, 
                'label_ids': label_ids,
                'input_lengths': input_length
            })
        
        return features    
    
    def _get_input_length(self, text, tokenizer):
        return len(text) if len(text) < tokenizer.max_seq_len else tokenizer.max_seq_len

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):
        
        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = vanilla_tokenizer.text_to_sequence(row_['text'])   
            label_ids = [self.cat2id[label_] for label_ in row_['label']]
            
            features.append({
                'input_ids': input_ids,
                'label_ids': label_ids
            })
        
        return features