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

from ark_nlp.model.ie.prompt_uie.utils import get_span
from ark_nlp.model.ie.prompt_uie.utils import get_bool_ids_greater_than


class PromptUIEPredictor(object):
    """
    通用信息抽取UIE的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
    """  # noqa: ignore flake8"

    def __init__(self, module, tokernizer):
        self.module = module
        self.module.task = 'TokenLevel'

        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

    def _convert_to_transfomer_ids(self, text, prompt):
        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        prompt_tokens = self.tokenizer.tokenize(prompt)

        input_ids = self.tokenizer.sequence_to_ids(prompt_tokens, tokens, truncation_method='last')
        input_ids, input_mask, segment_ids = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
        }

        return features, token_mapping

    def _get_input_ids(self, text, prompt):
        if self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text, prompt)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(self, features):
        return {
            col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device)
            for col in features
        }

    def predict_one_sample(
        self,
        text,
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
        """  # noqa: ignore flake8"

        text, prompt = text
        features, token_mapping = self._get_input_ids(text, prompt)

        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            start_logits, end_logits = self.module(**inputs)

            start_scores = start_logits[0].cpu().numpy()[2 + len(self.tokenizer.tokenize(prompt)):]
            end_scores = end_logits[0].cpu().numpy()[2 + len(self.tokenizer.tokenize(prompt)):]

            start_scores = get_bool_ids_greater_than(start_scores)
            end_scores = get_bool_ids_greater_than(end_scores)

        entities = []
        for span in get_span(start_scores, end_scores):

            if span[0] >= len(token_mapping) or span[-1] >= len(token_mapping):
                continue

            entitie_ = {
                "start_idx": token_mapping[span[0]][0],
                "end_idx": token_mapping[span[-1]][-1],
                "type": prompt,
                "entity": text[token_mapping[span[0]][0]:token_mapping[span[-1]][-1] + 1]
            }
            entities.append(entitie_)

        return entities
