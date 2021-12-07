"""
# Copyright 2021 Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

import torch

from torch.utils.data import DataLoader
from ark_nlp.factory.predictor.base._predictor import Predictor


class SequenceClassificationPredictor(Predictor):
    """
    序列分类任务的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):
        super(SequenceClassificationPredictor, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'SequenceLevel'

    def predict_one_sample(
        self,
        text='',
        topk=1,
        return_label_name=True,
        return_proba=False
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
            topk (:obj:`int`, optional, defaults to 1): 返回TopK结果
            return_label_name (:obj:`bool`, optional, defaults to True): 返回结果的标签ID转化成原始标签
            return_proba (:obj:`bool`, optional, defaults to False): 返回结果是否带上预测的概率
        """  # noqa: ignore flake8"

        if topk is None:
            topk = len(self.cat2id) if len(self.cat2id) > 2 else 1

        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs)
            logit = torch.nn.functional.softmax(logit, dim=1)

        probs, indices = logit.topk(topk, dim=1, sorted=True)

        preds = []
        probas = []
        for pred_, proba_ in zip(indices.cpu().numpy()[0], probs.cpu().numpy()[0].tolist()):

            if return_label_name:
                pred_ = self.id2cat[pred_]

            preds.append(pred_)

            if return_proba:
                probas.append(proba_)

        if return_proba:
            return list(zip(preds, probas))

        return preds

    def predict_batch(
        self,
        test_data,
        batch_size=16,
        shuffle=False,
        return_label_name=True,
        return_proba=False
    ):
        """
        batch样本预测

        Args:
            test_data (:obj:`ark_nlp dataset`): 输入batch文本
            batch_size (:obj:`int`, optional, defaults to 16): batch大小
            shuffle (:obj:`bool`, optional, defaults to False): 是否打扰数据集
            return_proba (:obj:`bool`, optional, defaults to False): 返回结果是否带上预测的概率
        """  # noqa: ignore flake8"

        self.inputs_cols = test_data.dataset_cols

        preds = []
        probas = []

        self.module.eval()
        generator = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for step, inputs in enumerate(generator):
                inputs = self._get_module_batch_inputs(inputs)

                logits = self.module(**inputs)

                preds.extend(torch.max(logits, 1)[1].cpu().numpy())
                if return_proba:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                    probas.extend(logits.max(dim=1).values.cpu().detach().numpy())

        if return_label_name:
            preds = [self.id2cat[pred_] for pred_ in preds]

        if return_proba:
            return list(zip(preds, probas))

        return preds
