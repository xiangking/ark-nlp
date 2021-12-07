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

from ark_nlp.dataset.base._dataset import BaseDataset


class SentenceClassificationDataset(BaseDataset):
    """
    用于序列分类任务的Dataset

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
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

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = bert_tokenizer.sequence_to_ids(row_['text'])

            input_ids, input_mask, segment_ids = input_ids

            feature = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            tokens = vanilla_tokenizer.tokenize(row_['text'])
            length = len(tokens)
            input_ids = vanilla_tokenizer.sequence_to_ids(tokens)

            feature = {
                'input_ids': input_ids,
                'length': length if length < vanilla_tokenizer.max_seq_len else vanilla_tokenizer.max_seq_len
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features


class PairMergeSentenceClassificationDataset(BaseDataset):
    """
    用于句子对合并后进行序列分类任务的Dataset，例如BERT分类任务

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
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

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = bert_tokenizer.sequence_to_ids(row_['text_a'], row_['text_b'])

            input_ids, input_mask, segment_ids = input_ids

            feature = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features


class TwinTowersSentenceClassificationDataset(BaseDataset):
    """
    用于双塔序列分类任务的Dataset，即句子对不组合，分开输入模型

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
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

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):

            input_ids_a = bert_tokenizer.sequence_to_ids(row_['text_a'])
            input_ids_b = bert_tokenizer.sequence_to_ids(row_['text_b'])

            input_ids_a, input_mask_a, segment_ids_a = input_ids_a
            input_ids_b, input_mask_b, segment_ids_b = input_ids_b

            feature = {
                'input_ids_a': input_ids_a,
                'attention_mask_a': input_mask_a,
                'token_type_ids_a': segment_ids_a,
                'input_ids_b': input_ids_b,
                'attention_mask_b': input_mask_b,
                'token_type_ids_b': segment_ids_b
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):

            input_ids_a = vanilla_tokenizer.sequence_to_ids(row_['text_a'])
            input_ids_b = vanilla_tokenizer.sequence_to_ids(row_['text_b'])

            feature = {
                'input_ids_a': input_ids_a,
                'input_ids_b': input_ids_b
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features
