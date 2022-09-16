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

from tqdm import tqdm
from ark_nlp.dataset.base._dataset import BaseDataset


class SentenceClassificationDataset(BaseDataset):
    """
    用于序列分类任务的Dataset

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
        return sorted(list(set([data['label'] for data in self.dataset])))

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text'] = data_df['text'].apply(lambda x: x.lower().strip())

        feature_names = list(data_df.columns)
        for index, row in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name: getattr(row, feature_name)
                for feature_name in feature_names
            })

        return dataset

    def _convert_to_transformer_ids(self, tokenizer):

        features = []
        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Converting sequence to transformer ids',
                )):
            input_ids = tokenizer.sequence_to_ids(row['text'])

            input_ids, attention_mask, token_type_ids = input_ids

            feature = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features

    def _convert_to_vanilla_ids(self, tokenizer):

        features = []
        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Converting sequence to  ids',
                )):
            tokens = tokenizer.tokenize(row['text'])
            input_ids = tokenizer.sequence_to_ids(tokens)

            feature = {
                'input_ids': input_ids,
                'sequence_length': len(tokens)
                if len(tokens) < tokenizer.max_seq_len else tokenizer.max_seq_len
            }

            if not self.is_test:
                label_ids = self.cat2id[row['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features


class PairMergeSentenceClassificationDataset(BaseDataset):
    """
    用于句子对合并后进行序列分类任务的Dataset, 例如BERT分类任务

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
        return sorted(list(set([data['label'] for data in self.dataset])))

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text_a'] = data_df['text_a'].apply(lambda x: x.lower().strip())
        data_df['text_b'] = data_df['text_b'].apply(lambda x: x.lower().strip())

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        return dataset

    def _convert_to_transformer_ids(self, tokenizer):

        features = []
        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Converting sequence to transformer ids',
                )):
            input_ids = tokenizer.sequence_to_ids(row['text_a'], row['text_b'])

            input_ids, attention_mask, token_type_ids = input_ids

            feature = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features


class PairWiseSentenceClassificationDataset(BaseDataset):
    """
    用于双塔序列分类任务的Dataset, 即句子对不组合, 分开输入模型

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
        return sorted(list(set([data['label'] for data in self.dataset])))

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text_a'] = data_df['text_a'].apply(lambda x: x.lower().strip())
        data_df['text_b'] = data_df['text_b'].apply(lambda x: x.lower().strip())

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        return dataset

    def _convert_to_transformer_ids(self, tokenizer):

        features = []
        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Converting sequence to transformer ids',
                )):

            input_ids_a = tokenizer.sequence_to_ids(row['text_a'])
            input_ids_b = tokenizer.sequence_to_ids(row['text_b'])

            input_ids_a, attention_mask_a, token_type_ids_a = input_ids_a
            input_ids_b, attention_mask_b, token_type_ids_b = input_ids_b

            feature = {
                'input_ids_a': input_ids_a,
                'attention_mask_a': attention_mask_a,
                'token_type_ids_a': token_type_ids_a,
                'input_ids_b': input_ids_b,
                'attention_mask_b': attention_mask_b,
                'token_type_ids_b': token_type_ids_b
            }

            if not self.is_test:
                label_ids = self.cat2id[row['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features

    def _convert_to_vanilla_ids(self, tokenizer):

        features = []
        for index, row in enumerate(
                tqdm(
                    self.dataset,
                    disable=not self.progress_verbose,
                    desc='Converting sequence to ids',
                )):

            input_ids_a = tokenizer.sequence_to_ids(row['text_a'])
            input_ids_b = tokenizer.sequence_to_ids(row['text_b'])

            feature = {'input_ids_a': input_ids_a, 'input_ids_b': input_ids_b}

            if not self.is_test:
                label_ids = self.cat2id[row['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features
