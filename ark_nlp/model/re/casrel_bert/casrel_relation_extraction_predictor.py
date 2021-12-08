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


class CasRelREPredictor(object):
    """
    CasRel bert模型的联合关系抽取任务的预测器

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
        tokens = self.tokenizer.tokenize(text)[:self.tokenizer.max_seq_len]
        token_mapping = self.tokenizer.get_token_mapping(text, tokens, is_mapping_index=False)

        input_ids, input_mask, segment_ids = self.tokenizer.sequence_to_ids(tokens)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_mapping': token_mapping
            }

        return features

    def _get_input_ids(
        self,
        text
    ):
        if self.tokenizer.tokenizer_type == 'transfomer':
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
        h_bar=0.5,
        t_bar=0.5
    ):

        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            encoded_text = self.module.bert(inputs['input_ids'], inputs['attention_mask'])[0]

            pred_sub_heads, pred_sub_tails = self.module.get_subs(encoded_text)
            sub_heads, sub_tails = np.where(pred_sub_heads.cpu()[0] > h_bar)[0], np.where(pred_sub_tails.cpu()[0] > t_bar)[0]

            token_mapping = inputs['token_mapping']

            subjects = []
            for sub_head in sub_heads:
                sub_tail = sub_tails[sub_tails >= sub_head]
                if len(sub_tail) > 0:
                    sub_tail = sub_tail[0]
                    subject = ''.join([token_mapping[index_] if index_ < len(token_mapping) else '' for index_ in range(sub_head-1, sub_tail)])
                    subjects.append((subject, sub_head, sub_tail))

            if subjects:
                triple_list = []
                repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                sub_head_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                sub_tail_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                for subject_idx, subject in enumerate(subjects):
                    sub_head_mapping[subject_idx][0][subject[1]] = 1
                    sub_tail_mapping[subject_idx][0][subject[2]] = 1
                sub_tail_mapping = sub_tail_mapping.to(repeated_encoded_text)
                sub_head_mapping = sub_head_mapping.to(repeated_encoded_text)

                pred_obj_heads, pred_obj_tails = self.module.get_objs_for_specific_sub(
                    sub_head_mapping,
                    sub_tail_mapping,
                    repeated_encoded_text
                )

                for subject_idx, subject in enumerate(subjects):
                    sub = subject[0]

                    obj_heads, obj_tails = np.where(pred_obj_heads.cpu()[subject_idx] > h_bar), np.where(pred_obj_tails.cpu()[subject_idx] > t_bar)
                    for obj_head, rel_head in zip(*obj_heads):
                        for obj_tail, rel_tail in zip(*obj_tails):
                            if obj_head <= obj_tail and rel_head == rel_tail:
                                rel = self.id2cat[int(rel_head)]
                                obj = ''.join([token_mapping[index_] if index_ < len(token_mapping) else '' for index_ in range(obj_head-1, obj_tail)])
                                triple_list.append((sub, rel, obj))
                                break
                triple_set = set()
                for s, r, o in triple_list:
                    if o == '' or s == '':
                        continue
                    triple_set.add((s, r, o))
                pred_list = list(triple_set)
            else:
                pred_list = []

        pred_triples = set(pred_list)

        return pred_triples
