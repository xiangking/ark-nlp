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

import copy
import numpy as np

import torch
from tqdm import tqdm
from collections import defaultdict
from ark_nlp.dataset.base._dataset import BaseDataset


class PFNREDataset(BaseDataset):
    """
    用于PRGC Bert联合关系抽取任务的Dataset

    Args:
        data (DataFrame or string): 数据或者数据地址
        categories (list or None, optional): 数据类别, 默认值为: None
        do_retain_df (bool, optional): 是否将DataFrame格式的原始数据复制到属性retain_df中, 默认值为: False
        do_retain_dataset (bool, optional): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中, 默认值为: False
        is_train (bool, optional): 数据集是否为训练集数据, 默认值为: True
        is_test (bool, optional): 数据集是否为测试集数据, 默认值为: False
        progress_verbose (bool, optional): 是否显示数据进度, 默认值为: True
    """  # noqa: ignore flake8"

    def __init__(self, data, ner_categories=None, *args, **kwargs):
        super(PFNREDataset, self).__init__(data, *args, **kwargs)

        if ner_categories is None:
            self.ner_categories = self._get_ner_categories()
        else:
            self.ner_categories = ner_categories

        self.ner2id = dict(zip(self.ner_categories, range(len(self.ner_categories))))
        self.id2ner = dict(zip(range(len(self.ner_categories)), self.ner_categories))

    def _get_categories(self):
        return sorted(
            list(set([triple[3] for data_ in self.dataset
                      for triple in data_['label']])))

    def _get_ner_categories(self):

        return ['None']

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text'] = data_df['text'].apply(lambda x: x.strip())
        if not self.is_test:
            data_df['label'] = data_df['label'].apply(lambda x: eval(x))

        feature_names = list(data_df.columns)
        for index, row in enumerate(data_df.itertuples()):

            dataset.append({
                feature_name: getattr(row, feature_name)
                for feature_name in feature_names
            })

        return dataset

    def _convert_to_transformer_ids(self, tokenizer):
        self.tokenizer = tokenizer

        if self.do_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)

        features = []
        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Converting sequence to transformer ids',
                )):

            text = row['text']

            if len(text) > self.tokenizer.max_seq_len - 2:
                text = text[:self.tokenizer.max_seq_len - 2]

            tokens = self.tokenizer.tokenize(text)
            token_mapping = self.tokenizer.get_token_mapping(text,
                                                             tokens,
                                                             is_mapping_index=False)
            index_token_mapping = self.tokenizer.get_token_mapping(text, tokens)

            start_mapping = {j[0]: i for i, j in enumerate(index_token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(index_token_mapping) if j}

            input_ids, attention_mask, _ = self.tokenizer.sequence_to_ids(tokens)

            ner_labels = torch.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len, len(self.ner2id)))
            re_head_labels = torch.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len, len(self.cat2id)))
            re_tail_labels = torch.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len, len(self.cat2id)))

            for triple in row['label']:
                sub_head_idx = triple[1]
                sub_end_idx = triple[2]
                obj_head_idx = triple[5]
                obj_end_idx = triple[6]

                if (sub_head_idx in start_mapping and obj_head_idx in start_mapping
                        and sub_end_idx in end_mapping
                        and obj_end_idx in end_mapping):
                    sub_head_idx = start_mapping[sub_head_idx]
                    sub_end_idx = end_mapping[sub_end_idx]
                    obj_head_idx = start_mapping[obj_head_idx]
                    obj_end_idx = end_mapping[obj_end_idx]

                    if sub_head_idx > sub_end_idx or obj_head_idx > obj_end_idx:
                        assert(1>2)
                        continue

                    ner_labels[sub_head_idx+1, sub_end_idx+1, self.ner2id["None"]] = 1
                    ner_labels[obj_head_idx+1, obj_end_idx+1, self.ner2id["None"]] = 1
                    re_head_labels[sub_head_idx+1, obj_head_idx+1, self.cat2id[triple[3]]] = 1
                    re_tail_labels[sub_end_idx+1, obj_end_idx+1, self.cat2id[triple[3]]] = 1

                feature = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'ner_labels': ner_labels.to_sparse(),
                    're_head_labels': re_head_labels.to_sparse(),
                    're_tail_labels': re_tail_labels.to_sparse(),
                    'token_mapping': token_mapping
                }

            features.append(feature)

        return features

    # @property
    # def to_device_cols(self):
    #     if self.is_train:
    #         return [
    #             'input_ids', 'attention_mask', 'ner_labels', 'rc_labels',
    #         ]
    #     else:
    #         return ['input_ids', 'attention_mask']