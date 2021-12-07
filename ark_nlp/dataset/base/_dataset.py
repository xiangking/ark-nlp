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

import json
import copy
import codecs
import pandas as pd

from collections import defaultdict
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame


class BaseDataset(Dataset):
    """
    Dataset基类

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def __init__(
        self,
        data,
        categories=None,
        is_retain_df=False,
        is_retain_dataset=False,
        is_train=True,
        is_test=False
    ):

        self.is_test = is_test
        self.is_train = is_train
        self.is_retain_df = is_retain_df
        self.is_retain_dataset = is_retain_dataset

        if self.is_test is True:
            self.is_train = False

        if isinstance(data, DataFrame):
            if 'label' in data.columns:
                data['label'] = data['label'].apply(lambda x: str(x))

            if self.is_retain_df:
                self.df = data

            self.dataset = self._convert_to_dataset(data)
        else:
            self.dataset = self._load_dataset(data)

        if categories is None:
            self.categories = self._get_categories()
        else:
            self.categories = categories

        if self.categories is not None:
            self.cat2id = dict(zip(self.categories, range(len(self.categories))))
            self.id2cat = dict(zip(range(len(self.categories)), self.categories))

            self.class_num = len(self.cat2id)

    def _get_categories(self):
        return None

    def _load_dataset(self, data_path):
        """
        加载数据集

        Args:
            data_path (:obj:`string`): 数据地址
        """  # noqa: ignore flake8"

        data_df = self._read_data(data_path)

        if self.is_retain_df:
            self.df = data_df

        return self._convert_to_dataset(data_df)

    def _convert_to_dataset(self, data_df):
        pass

    def _read_data(
        self,
        data_path,
        data_format=None,
        skiprows=-1
    ):
        """
        读取所需数据

        Args:
            data_path (:obj:`string`): 数据地址
            data_format (:obj:`string`, defaults to `None`): 数据存储格式
            skiprows (:obj:`int`, defaults to -1): 读取跳过指定行数，默认为不跳过
        """  # noqa: ignore flake8"

        if data_format is not None:
            data_format = data_path.split('.')[-1]

        if data_format == 'csv':
            data_df = pd.read_csv(data_path, dtype={'label': str})
        elif data_format == 'json':
            try:
                data_df = pd.read_json(data_path, dtype={'label': str})
            except:
                data_df = self.read_line_json(data_path)
        elif data_format == 'tsv':
            data_df = pd.read_csv(data_path, sep='\t', dtype={'label': str})
        elif data_format == 'txt':
            data_df = pd.read_csv(data_path, sep='\t', dtype={'label': str})
        else:
            raise ValueError("The data format does not exist")

        return data_df

    def read_line_json(
        self,
        data_path,
        skiprows=-1
    ):
        """
        读取所需数据

        Args:
            data_path (:obj:`string`): 数据所在路径
            skiprows (:obj:`int`, defaults to -1): 读取跳过指定行数，默认为不跳过
        """
        datasets = []

        with codecs.open(data_path, mode='r', encoding='utf8') as f:
            reader = f.readlines()
            for index, line in enumerate(reader):
                if index == skiprows:
                    continue
                line = json.loads(line)
                tokens = line['text']
                label = line['label']
                datasets.append({'text': tokens.strip(), 'label': label})

        return pd.DataFrame(datasets)

    def convert_to_ids(self, tokenizer):
        """
        将文本转化成id的形式

        Args:
            tokenizer: 编码器
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

    def _convert_to_transfomer_ids(self, bert_tokenizer):
        pass

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):
        pass

    def _convert_to_customized_ids(self, customized_tokenizer):
        pass

    def _get_input_length(self, text, bert_tokenizer):
        pass

    @property
    def dataset_cols(self):
        return list(self.dataset[0].keys())

    @property
    def to_device_cols(self):
        return list(self.dataset[0].keys())

    @property
    def sample_num(self):
        return len(self.dataset)

    @property
    def dataset_analysis(self):

        _result = defaultdict(list)
        for _row in self.dataset:
            for _col in self.dataset_cols:
                if type(_row[_col]) == str:
                    _result[_col].append(len(_row[_col]))

        _report = pd.DataFrame(_result).describe()

        return _report

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
