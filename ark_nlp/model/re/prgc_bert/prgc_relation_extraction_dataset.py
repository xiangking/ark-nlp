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
import numpy as np

from collections import defaultdict
from ark_nlp.dataset.base._dataset import BaseDataset


class PRGCREDataset(BaseDataset):
    """
    用于PRGC Bert联合关系抽取任务的Dataset

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):
        super(PRGCREDataset, self).__init__(*args, **kwargs)
        self.sublabel2id = {"B-H": 1, "I-H": 2, "O": 0}
        self.oblabel2id = {"B-T": 1, "I-T": 2, "O": 0}

    def _get_categories(self):
        return sorted(list(set([triple_[3] for data_ in self.dataset for triple_ in data_['label']])))

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text'] = data_df['text'].apply(lambda x: x.strip())
        if not self.is_test:
            data_df['label'] = data_df['label'].apply(lambda x: eval(x))

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):

            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        return dataset

    def _convert_to_transfomer_ids(self, tokenizer):
        self.tokenizer = tokenizer

        if self.is_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)

        features = []
        for (index_, row_) in enumerate(self.dataset):

            text = row_['text']

            if len(text) > self.tokenizer.max_seq_len - 2:
                text = text[:self.tokenizer.max_seq_len - 2]

            tokens = self.tokenizer.tokenize(text)
            token_mapping = self.tokenizer.get_token_mapping(text, tokens, is_mapping_index=False)
            index_token_mapping = self.tokenizer.get_token_mapping(text, tokens)

            start_mapping = {j[0]: i for i, j in enumerate(index_token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(index_token_mapping) if j}

            input_ids, input_mask, segment_ids = self.tokenizer.sequence_to_ids(tokens)

            if not self.is_train:
                triples = []

                for triple in row_['label']:
                    sub_head_idx = triple[1]
                    sub_end_idx = triple[2]
                    obj_head_idx = triple[5]
                    obj_end_idx = triple[6]

                    if sub_head_idx in start_mapping and obj_head_idx in start_mapping and sub_end_idx in end_mapping and obj_end_idx in end_mapping:
                        sub_head_idx = start_mapping[sub_head_idx]
                        obj_head_idx = start_mapping[obj_head_idx]

                        triples.append((('H', sub_head_idx + 1, end_mapping[sub_end_idx] + 1 + 1),
                                        ('T', obj_head_idx + 1, end_mapping[obj_end_idx] + 1 + 1),
                                        self.cat2id[triple[3]]))

                feature = {
                    'input_ids': input_ids,
                    'attention_mask': input_mask,
                    'triples': triples,
                    'token_mapping': token_mapping
                }

                features.append(feature)

            else:
                corres_tag = np.zeros((
                    self.tokenizer.max_seq_len,
                    self.tokenizer.max_seq_len
                ))

                rel_tag = len(self.cat2id) * [0]
                rel_entities = defaultdict(set)

                for triple in row_['label']:
                    sub_head_idx = triple[1]
                    sub_end_idx = triple[2]
                    obj_head_idx = triple[5]
                    obj_end_idx = triple[6]

                    # construct relation tag
                    rel_tag[self.cat2id[triple[3]]] = 1

                    if sub_head_idx in start_mapping and obj_head_idx in start_mapping and sub_end_idx in end_mapping and obj_end_idx in end_mapping:
                        sub_head_idx = start_mapping[sub_head_idx]
                        obj_head_idx = start_mapping[obj_head_idx]

                        corres_tag[sub_head_idx+1][obj_head_idx+1] = 1
                        rel_entities[self.cat2id[triple[3]]].add((sub_head_idx, end_mapping[sub_end_idx], obj_head_idx, end_mapping[obj_end_idx]))

                for rel, en_ll in rel_entities.items():
                    # init
                    tags_sub = self.tokenizer.max_seq_len * [self.sublabel2id['O']]
                    tags_obj = self.tokenizer.max_seq_len * [self.oblabel2id['O']]

                    for en in en_ll:
                        # get sub and obj head
                        sub_head_idx, sub_end_idx, obj_head_idx, obj_end_idx = en

                        tags_sub[sub_head_idx + 1] = self.sublabel2id['B-H']
                        tags_sub[sub_head_idx + 1 + 1: sub_end_idx + 1 + 1] = (sub_end_idx - sub_head_idx) * [self.sublabel2id['I-H']]

                        tags_obj[obj_head_idx + 1] = self.oblabel2id['B-T']
                        tags_obj[obj_head_idx + 1 + 1: obj_end_idx + 1 + 1] = (obj_end_idx - obj_head_idx) * [self.oblabel2id['I-T']]

                    seq_tag = [tags_sub, tags_obj]

                    feature = {
                        'input_ids': input_ids,
                        'attention_mask': input_mask,
                        'corres_tags': corres_tag,
                        'seq_tags': seq_tag,
                        'potential_rels': rel,
                        'rel_tags': rel_tag,
                        'token_mapping': token_mapping
                    }

                    features.append(feature)

        return features

    @property
    def to_device_cols(self):
        if self.is_train:
            return ['input_ids', 'attention_mask', 'corres_tags', 'seq_tags', 'potential_rels', 'rel_tags']
        else:
            return ['input_ids', 'attention_mask']
