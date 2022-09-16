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
import numpy as np

from ark_nlp.factory.utils.span_decode import get_entities_for_w2ner, convert_text_to_index


class W2NERPredictor(object):
    """
    W2NER的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (dict): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id,
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
        tokens = self.tokenizer.tokenize(text)[:self.tokenizer.max_seq_len - 2]

        input_ids = self.tokenizer.sequence_to_ids(tokens)
        input_ids, input_mask, segment_ids = input_ids

        # sequence_length 对应源码 sent_length
        sequence_length = len(tokens)
        _grid_mask2d = np.ones((sequence_length, sequence_length), dtype=np.bool)
        _dist_inputs = np.zeros((sequence_length, sequence_length), dtype=np.int)
        _pieces2word = np.zeros((sequence_length, sequence_length + 2), dtype=np.bool)

        # pieces2word 类似于token_mapping
        start = 0
        for i, pieces in enumerate(tokens):
            # 对齐源码
            pieces = [pieces]
            if len(pieces) == 0:
                continue
            pieces = list(range(start, start + len(pieces)))
            _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
            start += len(pieces)

        # dist_inputs
        # https://github.com/ljynlp/W2NER/issues/17
        dis2idx = np.zeros((1000), dtype='int64')
        dis2idx[1] = 1
        dis2idx[2:] = 2
        dis2idx[4:] = 3
        dis2idx[8:] = 4
        dis2idx[16:] = 5
        dis2idx[32:] = 6
        dis2idx[64:] = 7
        dis2idx[128:] = 8
        dis2idx[256:] = 9

        for k in range(sequence_length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(sequence_length):
            for j in range(sequence_length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        # 源码中 collate_fn 中处理成 max_lenth * max_lenth 矩阵代码
        def fill(data, new_data):
            new_data[:data.shape[0], :data.shape[1]] = torch.tensor(data, dtype=torch.long)
            return new_data

        mask2d_mat = torch.zeros((self.tokenizer.max_seq_len, self.tokenizer.max_seq_len))
        _grid_mask2d = fill(_grid_mask2d, mask2d_mat)
        dis_mat = torch.zeros((self.tokenizer.max_seq_len, self.tokenizer.max_seq_len))
        _dist_inputs = fill(_dist_inputs, dis_mat)
        sub_mat = torch.zeros((self.tokenizer.max_seq_len, self.tokenizer.max_seq_len))
        _pieces2word = fill(_pieces2word, sub_mat)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'grid_mask2d': _grid_mask2d,
            'dist_inputs': _dist_inputs,
            'pieces2word': _pieces2word,
            'sequence_length': sequence_length,
        }

        return features

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transformer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        tensors = dict()

        for col in features:
            if col == 'sequence_length':
                tensors[col] = torch.Tensor([features[col]])
            else:
                tensors[col] = torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device)

        return tensors

    def predict_one_sample(
            self,
            text=''
    ):
        """
        单样本预测

        Args:
            text (string): 输入文本
        """  # noqa: ignore flake8"

        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs)

        preds = torch.argmax(logit, -1)

        instance, l = preds.cpu().numpy()[0], int(inputs['sequence_length'].cpu().numpy()[0])

        predicts = get_entities_for_w2ner(instance, l)

        entities = []
        for entity_ in predicts:
            index, type = convert_text_to_index(entity_)
            entities.append({
                "idx": entity_,
                "entity": ''.join([text[i] for i in index]),
                "type": self.id2cat[type]
            })

        return entities
