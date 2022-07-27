# Copyright (c) 2022 DataArk Authors. All Rights Reserved.
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

import torch

from ark_nlp.dataset import TokenClassificationDataset


class PromptUIEDataset(TokenClassificationDataset):
    """
    用于通用信息抽取UIE任务的Dataset

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):

            prompt_tokens = bert_tokenizer.tokenize(row_['condition'])
            tokens = bert_tokenizer.tokenize(row_['text'])[:bert_tokenizer.max_seq_len - 3 - len(prompt_tokens)]
            token_mapping = bert_tokenizer.get_token_mapping(row_['text'], tokens)

            start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}

            input_ids = bert_tokenizer.sequence_to_ids(prompt_tokens, tokens, truncation_method='last')

            input_ids, input_mask, segment_ids = input_ids

            start_label = torch.zeros((bert_tokenizer.max_seq_len))
            end_label = torch.zeros((bert_tokenizer.max_seq_len))

            label_ = set()
            for info_ in row_['label']:
                if info_['start_idx'] in start_mapping and info_['end_idx'] in end_mapping:
                    start_idx = start_mapping[info_['start_idx']]
                    end_idx = end_mapping[info_['end_idx']]
                    if start_idx > end_idx or info_['entity'] == '':
                        continue

                    start_label[start_idx + 2 + len(prompt_tokens)] = 1
                    end_label[end_idx + 2 + len(prompt_tokens)] = 1

                    label_.add((start_idx + 2 + len(prompt_tokens),
                                end_idx + 2 + len(prompt_tokens)))

            features.append({
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids,
                'start_label_ids': start_label,
                'end_label_ids': end_label,
                'label_ids': list(label_)
            })

        return features

    @property
    def to_device_cols(self):
        _cols = list(self.dataset[0].keys())
        _cols.remove('label_ids')
        return _cols
