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

import copy
import random
import numpy as np

from ark_nlp.dataset.base._dataset import BaseDataset


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


class GPLinkerREDataset(BaseDataset):
    """
    用于GPLinker联合关系抽取任务的Dataset

    Args:
        data (DataFrame or string): 数据或者数据地址
        categories (list or None, optional): 数据类别, 默认值为: None
        do_retain_df (bool, optional): 是否将DataFrame格式的原始数据复制到属性retain_df中, 默认值为: False
        do_retain_dataset (bool, optional): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中, 默认值为: False
        is_train (bool, optional): 数据集是否为训练集数据, 默认值为: True
        is_test (bool, optional): 数据集是否为测试集数据, 默认值为: False
        progress_verbose (bool, optional): 是否显示数据进度, 默认值为: True
    """  # noqa: ignore flake8"

    def _get_categories(self):
        return sorted(
            list(set([triple[3] for data_ in self.dataset
                      for triple in data_['label']])))

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

    def convert_to_ids(self, tokenizer):
        """
        将文本转化成id的形式

        Args:
            tokenizer: 编码器

        ToDo: 将__getitem__部分ID化代码迁移到这部分
        """  # noqa: ignore flake8"

        self.tokenizer = tokenizer

        if self.do_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)

    def __getitem__(self, idx):
        ins_json_data = self.dataset[idx]
        text = ins_json_data['text']

        if len(text) > self.tokenizer.max_seq_len - 2:
            text = text[:self.tokenizer.max_seq_len - 2]

        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text,
                                                         tokens,
                                                         is_mapping_index=False)
        index_token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        start_mapping = {j[0]: i for i, j in enumerate(index_token_mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(index_token_mapping) if j}
        
        input_ids, attention_mask, token_type_ids = self.tokenizer.sequence_to_ids(tokens)

        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.categories))]
        tail_labels = [set() for i in range(len(self.categories))]
            
        for triple in ins_json_data['label']:
            sub_head_idx = triple[1]
            sub_end_idx = triple[2]
            obj_head_idx = triple[5]
            obj_end_idx = triple[6]
            relation_type = triple[3]
            
            if (sub_head_idx in start_mapping and obj_head_idx in start_mapping
                    and sub_end_idx in end_mapping and obj_end_idx in end_mapping):
                sub_head_idx = start_mapping[sub_head_idx]+1
                obj_head_idx = start_mapping[obj_head_idx]+1
                sub_end_idx = end_mapping[sub_end_idx]+1
                obj_end_idx = end_mapping[obj_end_idx]+1
            
                entity_labels[0].add((sub_head_idx, sub_end_idx))
                entity_labels[1].add((obj_head_idx, obj_end_idx))
                head_labels[self.cat2id[relation_type]].add((sub_head_idx, obj_head_idx))
                tail_labels[self.cat2id[relation_type]].add((sub_end_idx, obj_end_idx))
            
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))
                
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        
        return input_ids, attention_mask, token_type_ids, entity_labels, head_labels, tail_labels, ins_json_data['label'], token_mapping