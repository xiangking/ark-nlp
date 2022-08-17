# Copyright (c) 2020 DataArk Authors. All Rights Reserved.
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
        data (DataFrame or string): 数据或者数据地址
        categories (list or None, optional, defaults to `None`): 数据类别
        do_retain_df (bool, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        do_retain_dataset (bool, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (bool, optional, defaults to True): 数据集是否为训练集数据
        is_test (bool, optional, defaults to False): 数据集是否为测试集数据
        progress_verbose (bool, optional): 是否显示数据进度, 默认值为: True
    """  # noqa: ignore flake8"

    def __init__(self,
                 data,
                 categories=None,
                 do_retain_df=False,
                 do_retain_dataset=False,
                 is_train=True,
                 is_test=False,
                 progress_verbose=True):

        self.is_test = is_test
        self.is_train = is_train
        self.do_retain_df = do_retain_df
        self.do_retain_dataset = do_retain_dataset

        self.progress_verbose = progress_verbose

        if self.is_test is True:
            self.is_train = False

        if isinstance(data, DataFrame):
            if 'label' in data.columns:
                data['label'] = data['label'].apply(lambda x: str(x))

            if self.do_retain_df:
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
            data_path (string): 数据地址
        """  # noqa: ignore flake8"

        data_df = self._read_data(data_path)

        if self.do_retain_df:
            self.df = data_df

        return self._convert_to_dataset(data_df)

    def _convert_to_dataset(self, data_df):
        pass

    def _read_data(self, data_path, data_format=None, skiprows=-1):
        """
        读取所需数据

        Args:
            data_path (string): 数据地址
            data_format (string or None, optional): 数据存储格式, 默认值为None
            skiprows (int, optional): 读取跳过指定行数，默认为-1, 不跳过
        """  # noqa: ignore flake8"

        if data_format is None:
            data_format = data_path.split('.')[-1]

        if data_format == 'csv':
            data_df = pd.read_csv(data_path, dtype={'label': str})
        elif data_format == 'json':
            try:
                data_df = pd.read_json(data_path, dtype={'label': str})
            except Exception:
                data_df = self.read_line_json(data_path)
        elif data_format == 'tsv':
            data_df = pd.read_csv(data_path, sep='\t', dtype={'label': str})
        elif data_format == 'txt':
            data_df = pd.read_csv(data_path, sep='\t', dtype={'label': str})
        else:
            raise ValueError("The data format does not exist")

        return data_df

    def read_line_json(self, data_path, skiprows=-1):
        """
        读取所需数据

        Args:
            data_path (string): 数据地址
            skiprows (int, optional): 读取跳过指定行数，默认为-1, 不跳过
        """
        datasets = []

        with codecs.open(data_path, mode='r', encoding='utf8') as f:
            reader = f.readlines()
            for index, line in enumerate(reader):
                if index == skiprows:
                    continue
                datasets.append(json.loads(line))

        return pd.DataFrame(datasets)

    def convert_to_ids(self, tokenizer):
        """
        将文本转化成id的形式

        Args:
            tokenizer: 编码器
        """
        if tokenizer.tokenizer_type == 'vanilla':
            features = self._convert_to_vanilla_ids(tokenizer)
        elif tokenizer.tokenizer_type == 'transformer':
            features = self._convert_to_transformer_ids(tokenizer)
        elif tokenizer.tokenizer_type == 'customized':
            features = self._convert_to_customized_ids(tokenizer)
        else:
            raise ValueError("The tokenizer type does not exist")

        if self.do_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)

        self.dataset = features

    def _convert_to_transformer_ids(self, bert_tokenizer):
        pass

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):
        pass

    def _convert_to_customized_ids(self, customized_tokenizer):
        pass

    def _get_sequence_length(self, text, bert_tokenizer):
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
    def dataset_report(self):

        result = defaultdict(list)
        for row in self.dataset:
            for col_name in self.dataset_cols:
                if type(row[col_name]) == str:
                    result[col_name].append(len(row[col_name]))

        report_df = pd.DataFrame(result).describe()

        return report_df

    @property
    def max_text_length(self):

        records = dict()
        if 'text' in self.dataset[0]:
            records['text'] = max([len(row['text']) for row in self.dataset])
        if 'text_a' in self.dataset[0]:
            records['text_a'] = max([len(row['text_a']) for row in self.dataset])
        if 'text_b' in self.dataset[0]:
            records['text_b'] = max([len(row['text_b']) for row in self.dataset])

        return records

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
