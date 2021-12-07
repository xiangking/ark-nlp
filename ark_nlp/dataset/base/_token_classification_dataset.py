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


class TokenClassificationDataset(BaseDataset):
    """
    用于字符分类任务的Dataset

    Args:
        data (:obj:`DataFrame` or :obj:`string`): 数据或者数据地址
        categories (:obj:`list`, optional, defaults to `None`): 数据类别
        is_retain_df (:obj:`bool`, optional, defaults to False): 是否将DataFrame格式的原始数据复制到属性retain_df中
        is_retain_dataset (:obj:`bool`, optional, defaults to False): 是否将处理成dataset格式的原始数据复制到属性retain_dataset中
        is_train (:obj:`bool`, optional, defaults to True): 数据集是否为训练集数据
        is_test (:obj:`bool`, optional, defaults to False): 数据集是否为测试集数据
    """  # noqa: ignore flake8"

    def _convert_to_dataset(self, data_df):

        dataset = []

        data_df['text'] = data_df['text'].apply(lambda x: x.strip())
        if not self.is_test:
            data_df['label'] = data_df['label'].apply(
                lambda x: eval(x) if type(x) == str else x)

        feature_names = list(data_df.columns)
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({
                feature_name_: getattr(row_, feature_name_)
                for feature_name_ in feature_names
            })

        return dataset
