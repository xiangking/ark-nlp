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
from ark_nlp.dataset import BIOTokenClassificationDataset


class BiaffineNERDataset(BIOTokenClassificationDataset):
    def _get_categories(self):
        categories = sorted(list(set([label_['type'] for data in self.dataset for label_ in data['label']])))

        if 'O' in categories:
            categories.remove('O')
        categories.insert(0, 'O')
        return categories

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text'] = data_df['text'].apply(lambda x: x.lower().strip())
        if not self.is_test:
            data_df['label'] = data_df['label'].apply(lambda x: eval(x))

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({feature_name_: getattr(row_, feature_name_)
                             for feature_name_ in feature_names})

        return dataset

    def entity_search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            tokens = bert_tokenizer.tokenize(row_['text'])[:bert_tokenizer.max_seq_len-2]

            input_ids = bert_tokenizer.sequence_to_ids(tokens)

            input_ids, input_mask, segment_ids = input_ids

            zero = [0 for i in range(bert_tokenizer.max_seq_len)]
            span_mask = [input_mask for _ in range(sum(input_mask))]
            span_mask.extend([zero for _ in range(sum(input_mask),
                                                  bert_tokenizer.max_seq_len)])
            span_mask = np.array(span_mask)

            span_label = [0 for _ in range(bert_tokenizer.max_seq_len)]
            span_label = [span_label for _ in range(bert_tokenizer.max_seq_len)]
            span_label = np.array(span_label)

            for info_ in row_['label']:
                entity_tokens = bert_tokenizer.tokenize(info_['entity'])
                start_idx = self.entity_search(entity_tokens, tokens)
                if start_idx == -1:
                    continue
                end_idx = start_idx + len(entity_tokens) - 1

                if start_idx > end_idx:
                    continue

                span_label[start_idx+1, end_idx+1] = self.cat2id[info_['type']]

            features.append({
                'input_ids': input_ids, 
                'attention_mask': input_mask, 
                'token_type_ids': segment_ids, 
                'label_ids': span_label,
                'span_mask': span_mask
            })
        return features