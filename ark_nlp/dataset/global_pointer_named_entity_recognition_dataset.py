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

import torch

from ark_nlp.dataset import TokenClassificationDataset


class GlobalPointerNERDataset(TokenClassificationDataset):
    """
    用于GlobalPointer命名实体识别任务的Dataset

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def _get_categories(self):
        categories = sorted(list(set([label_['type'] for data in self.dataset for label_ in data['label']])))
        return categories

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            tokens = bert_tokenizer.tokenize(row_['text'])[:bert_tokenizer.max_seq_len-2]
            token_mapping = bert_tokenizer.get_token_mapping(row_['text'], tokens)

            start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}

            input_ids = bert_tokenizer.sequence_to_ids(tokens)

            input_ids, input_mask, segment_ids = input_ids

            global_label = torch.zeros((
                self.class_num,
                bert_tokenizer.max_seq_len,
                bert_tokenizer.max_seq_len)
            )

            for info_ in row_['label']:
                if info_['start_idx'] in start_mapping and info_['end_idx'] in end_mapping:
                    start_idx = start_mapping[info_['start_idx']]
                    end_idx = end_mapping[info_['end_idx']]
                    if start_idx > end_idx or info_['entity'] == '':
                        continue
                    global_label[self.cat2id[info_['type']], start_idx+1, end_idx+1] = 1

            features.append({
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'label_ids': global_label
            })

        return features
