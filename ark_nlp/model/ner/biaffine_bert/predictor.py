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

import torch


class BiaffineBertNERPredictor(object):
    """
    Biaffine命名实体的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (dict): 标签映射
    """  # noqa: ignore flake8"

    def __init__(self, module, tokenizer, cat2id):
        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokenizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat, index in self.cat2id.items():
            self.id2cat[index] = cat

        self.module.eval()

    def _convert_to_transformer_ids(self, text):
        tokens = self.tokenizer.tokenize(text)[:self.tokenizer.max_seq_len - 2]
        token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        input_ids = self.tokenizer.sequence_to_ids(tokens)
        input_ids, input_mask, segment_ids = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
        }

        return features, token_mapping

    def _get_input_ids(self, text):
        if self.tokenizer.tokenizer_type == 'transformer':
            return self._convert_to_transformer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(self, features):
        return {
            col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device)
            for col in features
        }

    def predict_one_sample(self, text=''):
        """
        单样本预测

        Args:
            text (string): 输入文本
        """  # noqa: ignore flake8"

        features, token_mapping = self._get_input_ids(text)

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            scores = torch.argmax(self.module(**inputs),
                                  dim=-1)[0].to(torch.device('cpu')).numpy().tolist()

        entities = []
        for start_idx in range(1, len(scores)):
            for end_idx in range(start_idx, len(scores[start_idx])):
                if scores[start_idx][end_idx] > 0:
                    if end_idx - 1 >= len(token_mapping):
                        break
                    if token_mapping[start_idx - 1][0] <= token_mapping[end_idx - 1][-1]:
                        entitie_ = {
                            "start_idx": token_mapping[start_idx - 1][0],
                            "end_idx": token_mapping[end_idx - 1][-1],
                            "entity": text[token_mapping[start_idx - 1][0]:token_mapping[end_idx - 1][-1] + 1],
                            "type": self.id2cat[scores[start_idx][end_idx]]
                        }

                        if entitie_['entity'] == '':
                            continue

                        entities.append(entitie_)

        return entities
