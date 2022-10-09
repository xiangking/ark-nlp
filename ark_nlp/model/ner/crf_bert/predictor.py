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

from ark_nlp.factory.utils.span_decode import get_entities


class CrfBertNERPredictor(object):
    """
    +CRF模式的字符分类任务的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (dict): 标签映射
    """  # noqa: ignore flake8"

    def __init__(self, module, tokernizer, cat2id, markup='bio'):
        self.markup = markup

        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

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
            logits = self.module(**inputs)

        tags = self.module.crf.decode(logits, inputs['attention_mask'])
        tags = tags.squeeze(0)

        preds = tags.cpu().numpy().tolist()[0][:inputs['attention_mask'].cpu().numpy().
                                               sum()]
        preds = preds[1:-1]

        tags = []
        for index, tag in enumerate(preds):

            if index >= len(token_mapping):
                break

            token_start_idx = token_mapping[index][0]
            token_end_idx = token_mapping[index][-1]

            if token_start_idx > 0 and token_start_idx != token_mapping[index - 1][-1] + 1:
                if self.id2cat[tag].split('-')[0] == 'I':
                    tags.append(self.id2cat[tag])
                else:
                    tags.append(self.id2cat[0])

            for _ in range(token_start_idx, token_end_idx + 1):
                tags.append(self.id2cat[tag])

        label_entities = get_entities(tags, self.id2cat, self.markup)

        entities = set()
        for entity_ in label_entities:
            entities.add(text[entity_[1]:entity_[2] + 1] + '-' + entity_[0])

        entities = []
        for entity_ in label_entities:
            entities.append({
                "start_idx": entity_[1],
                "end_idx": entity_[2],
                "entity": text[entity_[1]:entity_[2] + 1],
                "type": entity_[0]
            })

        return entities
