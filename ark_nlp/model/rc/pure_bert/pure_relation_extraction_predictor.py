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
# Author: Chenjie Shen, jimme.shen123@gmail.com
# Status: Active

import torch
import numpy as np


class PUREREPredictor(object):
    """
    AFEA bert模型的联合关系抽取任务的预测器

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
        self.module.task = 'SequenceLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text,
            entities
    ):
        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens, is_mapping_index=False)
        index_token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        start_mapping = {j[0]: i for i, j in enumerate(index_token_mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(index_token_mapping) if j}

        # 删去分词错误和超出最大长度的实体
        eidxs_ = []
        num = 0
        for eidx, entity in enumerate(entities):
            if entity[2] in start_mapping and entity[3] in end_mapping:
                if self.tokenizer.max_seq_len - 3 - (num + 1) * 2 < end_mapping[entity[3]]:
                    break
                else:
                    eidxs_.append(eidx)
                    num += 1

        entities_ = [entities[i] for i in eidxs_]

        if len(tokens) > (self.tokenizer.max_seq_len - 3 - len(entities_) * 2):
            tokens = tokens[0:(self.tokenizer.max_seq_len - 3 - len(entities_) * 2)]

        # 生成position_ids
        position_ids = [i for i in range(len(tokens) + 2)]
        tokens += ['[SEP]']

        entity2marker = {}
        idx = len(tokens)
        for entity in entities_:
            head_idx, end_idx = entity[2], entity[3]
            tokens += [f'[{entity[1]}]']
            tokens += [f'[/{entity[1]}]']
            position_ids.append(start_mapping[head_idx] + 1)
            position_ids.append(end_mapping[end_idx] + 1)
            entity2marker[head_idx] = idx + 1
            entity2marker[end_idx] = idx + 2
            idx = idx + 2

        pad = [0] * (self.tokenizer.max_seq_len - len(position_ids))
        position_ids.extend(pad)

        input_ids, input_mask, segment_ids = self.tokenizer.sequence_to_ids(tokens)

        feature = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'position_ids': position_ids,
        }

        corres_tag = np.ones((
            self.tokenizer.max_seq_len,
            self.tokenizer.max_seq_len
        )) * self.cat2id['None']

        relations_idx = []
        entity_pair = []
        label_ids = []
        for eidx_sub, sub in zip(eidxs_, entities_):
            for eidx_obj, obj in zip(eidxs_, entities_):
                if str(sub) == str(obj):
                    continue
                sub_head_idx = entity2marker[sub[2]]
                sub_end_idx = entity2marker[sub[3]]
                obj_head_idx = entity2marker[obj[2]]
                obj_end_idx = entity2marker[obj[3]]

                relations_idx.append([sub_head_idx, sub_end_idx, obj_head_idx, obj_end_idx])
                entity_pair.append([eidx_sub, eidx_obj])
                label_ids.append(corres_tag[sub_head_idx][obj_head_idx])

        feature['entity_pair'] = entity_pair
        feature['relations_idx'] = relations_idx

        return feature

    def _get_input_ids(
            self,
            text,
            entities,
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text, entities)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        tensors = {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}
        return tensors

    def predict_one_sample(
            self,
            text='',
            entities=list(),
            topk=1,
            return_label_name=True,
            return_proba=False
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
            topk (:obj:`int`, optional, defaults to 1): 返回TopK结果
            return_label_name (:obj:`bool`, optional, defaults to True): 返回结果的标签ID转化成原始标签
            return_proba (:obj:`bool`, optional, defaults to False): 返回结果是否带上预测的概率
        """  # noqa: ignore flake8"

        if topk is None:
            topk = len(self.cat2id) if len(self.cat2id) > 2 else 1

        features = self._get_input_ids(text, entities)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit, entity_pair = self.module(**inputs)
            logit = torch.nn.functional.softmax(logit, dim=1)

        probs, indices = logit.topk(topk, dim=1, sorted=True)

        preds = []
        probas = []
        for prob, indice, entity_pair_ in zip(probs, indices, entity_pair[0]):

            for pred_, proba_ in zip(indice.cpu().numpy(), prob.cpu().numpy().tolist()):

                if return_label_name:
                    pred_ = self.id2cat[pred_]

                preds.append([entities[entity_pair_[0]], pred_, entities[entity_pair_[1]]])

                if return_proba:
                    probas.append(proba_)

        if return_proba:
            return list(zip(preds, probas))

        return preds
