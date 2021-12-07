"""
# Copyright 2020 Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

import torch
import numpy as np

from ark_nlp.factory.utils.conlleval import get_entities
from ark_nlp.factory.predictor.base._sequence_classification import SequenceClassificationPredictor


class TokenClassificationPredictor(SequenceClassificationPredictor):
    """
    字符分类任务的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):

        super(TokenClassificationPredictor, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'TokenLevel'

    def predict_one_sample(
        self,
        text=''
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
        """  # noqa: ignore flake8"

        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs)

        preds = logit.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        preds = preds[0][1:]
        preds = preds[:len(text)]

        # tags = [self.id2cat[x] for x in preds]
        label_entities = get_entities(preds, self.id2cat, self.markup)

        entities = set()
        for entity_ in label_entities:
            entities.add(text[entity_[1]: entity_[2]+1] + '-' + entity_[0])

        entities = []
        for entity_ in label_entities:
            entities.append({
                "start_idx": entity_[1],
                "end_idx": entity_[2],
                "entity": text[entity_[1]: entity_[2]+1],
                "type": entity_[0]
            })

        return entities
