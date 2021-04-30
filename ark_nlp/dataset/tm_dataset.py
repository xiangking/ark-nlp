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
from ._dataset import BaseDataset


class TMDataset(BaseDataset):
    def __init__(
        self,
        data_path, 
        categories=None, 
        is_retain_dataset=False
    ):
        super(TMDataset, self).__init__(data_path, categories, is_retain_dataset)
        
    def _get_categories(self):
        return sorted(list(set([data['label'] for data in self.dataset])))
    
    def _convert_to_dataset(self, data_df):
        
        dataset = []
        
        data_df['text_a'] = data_df['text_a'].apply(lambda x: x.lower().strip())
        data_df['text_b'] = data_df['text_b'].apply(lambda x: x.lower().strip())
        
        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({feature_name_: getattr(row_, feature_name_) 
                             for feature_name_ in feature_names})
            
        return dataset

    def _convert_to_transfomer_ids(self, bert_tokenizer):
        
        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = bert_tokenizer.sequence_to_ids(row_['text_a'], row_['text_b'])
            
            input_ids, input_mask, segment_ids = input_ids
            
            label_ids = self.cat2id[row_['label']]
            
            input_a_length = self._get_input_length(row_['text_a'], bert_tokenizer)
            input_b_length = self._get_input_length(row_['text_b'], bert_tokenizer)
            
            features.append({
                'input_ids': input_ids, 
                'attention_mask': input_mask, 
                'token_type_ids': segment_ids, 
                'label_ids': label_ids
            })
        
        return features        

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):
        
        features = []
        for (index_, row_) in enumerate(self.dataset):

            input_a_ids = vanilla_tokenizer.sequence_to_ids(row_['text_a'])
            input_b_ids = vanilla_tokenizer.sequence_to_ids(row_['text_b'])   

            label_ids = self.cat2id[row_['label']]
            
            features.append({
                'input_a_ids': input_a_ids,
                'input_b_ids': input_b_ids,
                'label_ids': label_ids
            })
        
        return features