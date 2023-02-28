# Copyright (c) 2021 DataArk Authors. All Rights Reserved.
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

from torch.utils.data import DataLoader
from ark_nlp.factory.predictor import SequenceClassificationPredictor


class SupervisedSimCSEPredictor(SequenceClassificationPredictor):
    """
    SupervisedSimCSE的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(self, module, tokenizer, cat2id=None):
        self.module = module

        self.cat2id = cat2id
        if self.cat2id is None:
            self.cat2id = {'0': 0, '1': 1}

        self.tokenizer = tokenizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat, index in self.cat2id.items():
            self.id2cat[index] = cat

        self.module.eval()

    def _get_input_ids(self, text_a, text_b):
        if self.tokenizer.tokenizer_type == 'transformer':
            return self._convert_to_transformer_ids(text_a, text_b)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text_a, text_b)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _convert_to_transformer_ids(self, text_a, text_b):
        input_ids_a = self.tokenizer.sequence_to_ids(text_a)
        input_ids_b = self.tokenizer.sequence_to_ids(text_b)

        input_ids_a, input_mask_a, segment_ids_a = input_ids_a
        input_ids_b, input_mask_b, segment_ids_b = input_ids_b

        features = {
            'input_ids': input_ids_a,
            'attention_mask': input_mask_a,
            'token_type_ids': segment_ids_a,
            'contrastive_input_ids': input_ids_b,
            'contrastive_attention_mask': input_mask_b,
            'contrastive_token_type_ids': segment_ids_b
        }

        return features

    def predict_one_sample(self,
                           text,
                           threshold=0.5,
                           return_label_name=True,
                           return_proba=False):
        text_a, text_b = text
        features = self._get_input_ids(text_a, text_b)

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logits = self.module.cosine_sim(**inputs).cpu().numpy()

        _proba = logits[0]

        if threshold is not None:
            _pred = self._threshold(_proba, threshold)

        if return_label_name and threshold is not None:
            _pred = self.id2cat[_pred]

        if threshold is not None:
            if return_proba:
                return [_pred, _proba]
            else:
                return _pred

        return _proba

    def predict_batch(self, test_data, batch_size=16, shuffle=False):
        self.inputs_cols = test_data.dataset_cols

        preds = []

        generator = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

        with torch.no_grad():
            for step, inputs in enumerate(generator):
                inputs = self._get_module_batch_inputs(inputs)

                logits = self.module.cosine_sim(**inputs).cpu().numpy()

                preds.extend(logits)

        return preds