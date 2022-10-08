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


from ark_nlp.factory.predictor.base import SequenceClassificationPredictor


class TMPredictor(SequenceClassificationPredictor):
    """
    文本匹配任务的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (dict): 标签映射
    """  # noqa: ignore flake8"

    def _convert_to_transformer_ids(self, text_a, text_b):
        input_ids = self.tokenizer.sequence_to_ids(text_a, text_b)
        input_ids, input_mask, segment_ids = input_ids

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids
        }
        return features

    def _convert_to_vanilla_ids(self, text_a, text_b):
        input_ids_a = self.tokenizer.sequence_to_ids(text_a)
        input_ids_b = self.tokenizer.sequence_to_ids(text_b)

        features = {'input_ids_a': input_ids_a, 'input_ids_b': input_ids_b}
        return features

    def _get_input_ids(self, text):

        text_a, text_b = text

        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text_a, text_b)
        elif self.tokenizer.tokenizer_type == 'transformer':
            return self._convert_to_transformer_ids(text_a, text_b)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text_a, text_b)
        else:
            raise ValueError("The tokenizer type does not exist")
