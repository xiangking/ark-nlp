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
import numpy as np


class BiaffineNERPredictor(object):
    """
    Biaffine命名实体的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
        self,
        module,
        tokernizer,
        cat2id
    ):
        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
        self,
        text
    ):
        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        input_ids = self.tokenizer.sequence_to_ids(tokens)
        input_ids, input_mask, segment_ids = input_ids

        zero = [0 for i in range(self.tokenizer.max_seq_len)]
        span_mask = [input_mask for i in range(sum(input_mask))]
        span_mask.extend([zero for i in range(sum(input_mask), self.tokenizer.max_seq_len)])
        span_mask = np.array(span_mask)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'span_mask': span_mask
        }

        return features, token_mapping

    def _get_input_ids(
        self,
        text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
        self,
        features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
        self,
        text=''
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
        """  # noqa: ignore flake8"

        features, token_mapping = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            scores = torch.argmax(self.module(**inputs), dim=-1)[0].to(torch.device('cpu')).numpy().tolist()

        entities = []
        for start in range(len(scores)):
            for end in range(start, len(scores[start])):
                if scores[start][end] > 0:
                    if end-1 > token_mapping[-1][-1]:
                        break
                    if token_mapping[start-1][0] <= token_mapping[end-1][-1]:
                        entitie_ = {
                            "start_idx": token_mapping[start-1][0],
                            "end_idx": token_mapping[end-1][-1],
                            "entity": text[token_mapping[start-1][0]: token_mapping[end-1][-1]+1],
                            "type": self.id2cat[scores[start][end]]
                        }

                        if entitie_['entity'] == '':
                            continue

                        entities.append(entitie_)

        return entities
