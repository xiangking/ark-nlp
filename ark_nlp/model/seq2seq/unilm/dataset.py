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
# Author: Chenjie Shen, jimme.shen123@gmail.com
# Status: Active


import copy
import numpy as np
from tqdm import tqdm
from ark_nlp.dataset.base._dataset import BaseDataset


class UniLMDataset(BaseDataset):
    """
    用于Seq2Seq任务的Dataset

    Args:
        data (DataFrame or string): 数据或者数据地址
        categories (list or None, optional): 数据类别, 默认值为: None
        is_retain_df (bool, optional): 是否将DataFrame格式的原始数据复制到属性retain_df中, 默认值为: False
        is_retain_dataset (bool, optional): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中, 默认值为: False
        is_train (bool, optional): 数据集是否为训练集数据, 默认值为: True
        is_test (bool, optional): 数据集是否为测试集数据, 默认值为: False
    """  # noqa: ignore flake8"

    def _get_categories(self):
        return sorted(list(set([data['label'] for data in self.dataset])))

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text'] = data_df['text'].apply(lambda x: x.lower().strip())

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        return dataset

    def _convert_to_transformer_ids(self, tokenizer):

        self.tokenizer = tokenizer

        if self.do_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)

        self.keep_token = True
        self.keep_tokens = []

        features = []
        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Convert to transformer ids',
                )):

            text, label = row['text'], row['label']

            text_tokens = self.tokenizer.tokenize(text)
            label_tokens = self.tokenizer.tokenize(label)
            input_ids = self.tokenizer.sequence_to_ids(text_tokens, label_tokens)

            input_ids, attention_mask, token_type_ids = input_ids

            text_ids = self.tokenizer.vocab.convert_tokens_to_ids(['[CLS]'] + text_tokens + ['[SEP]'])
            text_token_type_ids = [0] * len(text_ids)

            if len(text_ids) > (self.tokenizer.max_seq_len - 3) // 2:
                text_ids = text_ids[:(self.tokenizer.max_seq_len - 3) // 2]
                text_token_type_ids = text_token_type_ids[:(self.tokenizer.max_seq_len - 3) // 2]

            feature = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'text_ids': text_ids,
                'text_token_type_ids': text_token_type_ids,
                'label': label
            }

            # TODO: 缩减字典
            if self.keep_token and self.is_train:
                self.keep_tokens.extend(input_ids)

            features.append(feature)

        self.keep_tokens = list(set(self.keep_tokens))

        return features