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

import numpy as np

from tqdm import tqdm
from ark_nlp.dataset import TokenClassificationDataset


class BiaffineBertNERDataset(TokenClassificationDataset):
    """
    用于Biaffine命名实体识别任务的Dataset

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
        categories = sorted(
            list(set([label['type'] for data in self.dataset
                      for label in data['label']])))
        if 'O' in categories:
            categories.remove('O')
        categories.insert(0, 'O')
        return categories

    def _convert_to_transformer_ids(self, tokenizer):

        features = []
        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Converting sequence to transformer ids',
                )):

            tokens = tokenizer.tokenize(row['text'])[:tokenizer.max_seq_len - 2]
            token_mapping = tokenizer.get_token_mapping(row['text'], tokens)

            start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}

            input_ids = tokenizer.sequence_to_ids(tokens)

            input_ids, attention_mask, token_type_ids = input_ids

            zero = [0 for _ in range(tokenizer.max_seq_len)]
            span_mask = [attention_mask for _ in range(sum(attention_mask))]
            span_mask.extend(
                [zero for _ in range(sum(attention_mask), tokenizer.max_seq_len)])
            span_mask = np.array(span_mask)

            span_label = [0 for _ in range(tokenizer.max_seq_len)]
            span_label = [span_label for _ in range(tokenizer.max_seq_len)]
            span_label = np.array(span_label)

            for info in row['label']:
                if info['start_idx'] in start_mapping and info['end_idx'] in end_mapping:
                    start_idx = start_mapping[info['start_idx']]
                    end_idx = end_mapping[info['end_idx']]
                    if start_idx > end_idx or info['entity'] == '':
                        continue

                    span_label[start_idx + 1, end_idx + 1] = self.cat2id[info['type']]

            features.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'label_ids': span_label,
                'span_mask': span_mask
            })

        return features
