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
import numpy as np


class GPLinkerREPredictor(object):
    """
    GPLinker模型的联合关系抽取任务的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (dict): 标签映射
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
        tokens = self.tokenizer.tokenize(text)[:self.tokenizer.max_seq_len]
        token_mapping = self.tokenizer.get_token_mapping(text, tokens, is_mapping_index=False)

        input_ids, input_mask, segment_ids = self.tokenizer.sequence_to_ids(tokens)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'token_mapping': token_mapping
            }

        return features

    def _get_input_ids(
        self,
        text
    ):
        if self.tokenizer.tokenizer_type == 'transformer':
            return self._convert_to_transfomer_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
        self,
        features
    ):
        inputs = {}
        for col in features:
            if isinstance(features[col], np.ndarray):
                inputs[col] = torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device)
            else:
                inputs[col] = features[col]

        return inputs

    def predict_one_sample(
        self,
        text='',
        threshold=0.0
    ):

        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            outputs = self.module(**inputs)
            
            token_mapping = inputs['token_mapping']

            entity_logits, head_logits, tail_logits = [output[0].cpu().numpy() for output in outputs]
                
            subjects, objects = set(), set()
            outputs[0][:, [0, -1]] -= np.inf
            outputs[0][:, :, [0, -1]] -= np.inf
            
            for entity_type, head_idx, tail_idx in zip(*np.where(entity_logits > 0)):
                if entity_type == 0:
                    subjects.add((head_idx, tail_idx))
                else:
                    objects.add((head_idx, tail_idx))
        
            pred_triples = set()
            for subject_head_idx, subject_tail_idx in subjects:
                for object_head_idx, object_tail_idx in objects:
                    head_relation_set = np.where(head_logits[:, subject_head_idx, object_head_idx] > threshold)[0]
                    tail_relation_set = np.where(tail_logits[:, subject_tail_idx, object_tail_idx] > threshold)[0]
                    relation_set = set(head_relation_set) & set(tail_relation_set)
                    for relation_type in relation_set:
                        subject = ''.join([token_mapping[index_] if index_ < len(token_mapping) else '' for index_ in range(subject_head_idx-1, subject_tail_idx)])
                        obj = ''.join([token_mapping[index_] if index_ < len(token_mapping) else '' for index_ in range(object_head_idx-1, object_tail_idx)])
                        pred_triples.add((
                            subject, 
                            self.id2cat[relation_type],
                            obj
                        ))
                        
        return list(pred_triples)