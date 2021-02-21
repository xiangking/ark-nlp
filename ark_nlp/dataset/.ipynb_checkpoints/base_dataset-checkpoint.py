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


class BaseDataset(Dataset):
    def __init__(self, data_path, categories=None, is_retain_dataset=True):
        self.dataset = self._load_dataset(data_path)
        
        if categories == None:
            self.categories = self._get_categories()
        else:
            self.categories = categories

        self.cat2id = dict(zip(self.categories, range(len(self.categories))))
        self.id2cat = dict(zip(range(len(self.categories)), self.categories))
        
        self.class_num = len(self.cat2id) 
        
        self.is_retain_dataset = is_retain_dataset
                
    def _get_categories(self):
        pass
            
    def _read_data(self, data_path, data_format=None):
        """
        读取所需数据
        
        :param data_path: (string) 数据所在路径
        :param data_format: (string) 数据存储格式
        """  
        
        if data_format == None:
            data_format = data_path.split('.')[-1]
        
        if data_format == 'csv':
            data_df = pd.read_csv(data_path)
        elif data_format == 'json':
            data_df = pd.read_json(data_path)
        elif data_format == 'tsv':
            data_df = pd.read_csv(data_path, sep='\t')
        else:
            raise ValueError("The data format does not exist") 
        
        return data_df
    
    def _convert_to_dataset(self, data_df):
        pass
        
    def _load_dataset(self, data_path):
        """
        加载数据集
        
        :param data_path: (string) the data file to load
        """
        data_df = self._read_data(data_path)
        
        return self._convert_to_dataset(data_df)
                
    def _get_input_length(self, text, bert_tokenizer):
        pass 
    
    def _convert_to_transfomer_ids(self, bert_tokenizer):
        pass

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):
        pass
    
    def _convert_to_customized_ids(self, customized_tokenizer):
        pass

    def convert_to_ids(self, tokenizer):
        """
        将文本转化成id的形式
        
        :param tokenizer: 
        """        
        if tokenizer.tokenizer_type == 'vanilla':
            features = self._convert_to_vanilla_ids(tokenizer)
        elif tokenizer.tokenizer_type == 'transfomer':
            features = self._convert_to_transfomer_ids(tokenizer)
        elif tokenizer.tokenizer_type == 'customized':
            features = self._convert_to_customized_ids(tokenizer)
        else:
            raise ValueError("The tokenizer type does not exist") 
            
        if self.is_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)
            
        self.dataset = features
    
    @property
    def dataset_cols(self):
        return list(self.dataset[0].keys())
    
    @property
    def sample_num(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)