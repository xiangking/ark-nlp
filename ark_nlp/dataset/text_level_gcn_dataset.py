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

from ark_nlp.dataset import SentenceClassificationDataset


class TextLevelGCNDataset(SentenceClassificationDataset):
    """
    用于TextLevelGCN文本分类任务的Dataset

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def convert_to_ids(self, tokenizer):
        """
        将文本转化成id的形式

        :param tokenizer:
        """
        if tokenizer.tokenizer_type == 'graph':
            features = self._convert_to_graph_ids(tokenizer)
        else:
            raise ValueError("The tokenizer type does not exist")

        if self.is_retain_dataset:
            self.retain_dataset = copy.deepcopy(self.dataset)

        self.dataset = features

    def _convert_to_graph_ids(self, graph_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            node_ids, edge_ids, sub_graph = graph_tokenizer.sequence_to_graph(row_['text'])

            feature = {
                'node_ids': node_ids,
                'edge_ids': edge_ids,
                'sub_graph': sub_graph
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features
