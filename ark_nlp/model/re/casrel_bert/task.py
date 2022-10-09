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

from collections import defaultdict
from ark_nlp.factory.metric import TripleMetric
from ark_nlp.factory.task.base import SequenceClassificationTask


def to_tup(triple_list):
    ret = []
    for triple in triple_list:
        ret.append(tuple([triple[1], triple[2], triple[3], triple[5], triple[6]]))
    return ret


class CasRelRETask(SequenceClassificationTask):
    """
    基于CasRel Bert的联合关系抽取任务的Task
    
    Args:
        module: 深度学习模型
        optimizer (str or torch.optim.Optimizer or None, optional): 训练模型使用的优化器名或者优化器对象, 默认值为: None
        loss_function (str or object or None, optional): 训练模型使用的损失函数名或损失函数对象, 默认值为: None
        scheduler (torch.optim.lr_scheduler.LambdaLR, optional): scheduler对象, 默认值为: None
        tokenizer (object or None, optional): 分词器, 默认值为: None
        class_num (int or None, optional): 标签数目, 默认值为: None
        gpu_num (int, optional): GPU数目, 默认值为: 1
        device (torch.device, optional): torch.device对象, 当device为None时, 会自动检测是否有GPU
        cuda_device (int, optional): GPU编号, 当device为None时, 根据cuda_device设置device, 默认值为: 0
        ema_decay (int or None, optional): EMA的加权系数, 默认值为: None
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):

        super(CasRelRETask, self).__init__(*args, **kwargs)

        if 'metric' not in kwargs:
            self.metric = TripleMetric()

    def _train_collate_fn(self, batch):
        return self.casrel_collate_fn(batch)

    def _evaluate_collate_fn(self, batch):
        return self.casrel_collate_fn(batch)

    def casrel_collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        batch.sort(key=lambda x: x[2], reverse=True)
        token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens, token_mapping = zip(
            *batch)
        cur_batch = len(batch)
        max_text_len = max(text_len)
        batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
        batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
        batch_sub_heads = torch.Tensor(cur_batch, max_text_len).zero_()
        batch_sub_tails = torch.Tensor(cur_batch, max_text_len).zero_()
        batch_sub_head = torch.Tensor(cur_batch, max_text_len).zero_()
        batch_sub_tail = torch.Tensor(cur_batch, max_text_len).zero_()
        batch_obj_heads = torch.Tensor(cur_batch, max_text_len, self.class_num).zero_()
        batch_obj_tails = torch.Tensor(cur_batch, max_text_len, self.class_num).zero_()

        for i in range(cur_batch):
            batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
            batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
            batch_sub_heads[i, :text_len[i]].copy_(torch.from_numpy(sub_heads[i]))
            batch_sub_tails[i, :text_len[i]].copy_(torch.from_numpy(sub_tails[i]))
            batch_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sub_head[i]))
            batch_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sub_tail[i]))
            batch_obj_heads[i, :text_len[i], :].copy_(torch.from_numpy(obj_heads[i]))
            batch_obj_tails[i, :text_len[i], :].copy_(torch.from_numpy(obj_tails[i]))

        return {
            'input_ids': batch_token_ids,
            'attention_mask': batch_masks,
            'sub_heads': batch_sub_heads,
            'sub_tails': batch_sub_tails,
            'sub_head': batch_sub_head,
            'sub_tail': batch_sub_tail,
            'obj_heads': batch_obj_heads,
            'obj_tails': batch_obj_tails,
            'label_ids': triples,
            'tokens': tokens,
            'token_mapping': token_mapping
        }

    def get_module_inputs_on_train(self, inputs, **kwargs):
        """模型输入处理方法"""
        for col in inputs.keys():
            if type(inputs[col]) is torch.Tensor:
                inputs[col] = inputs[col].to(self.device)

        return inputs

    def get_module_inputs_on_evaluate(self, inputs, **kwargs):
        for col in inputs.keys():
            if type(inputs[col]) is torch.Tensor:
                inputs[col] = inputs[col].to(self.device)

        return inputs

    def get_train_loss(self, inputs, outputs, **kwargs):
        # 计算损失
        loss = self.compute_loss(inputs, outputs, **kwargs)

        return outputs, loss

    def compute_loss(self, inputs, logits, **kwargs):

        loss = self.loss_function(logits, inputs)

        return loss

    def evaluate(self, validation_data, **kwargs):
        """
        验证方法
        
        Args:
            validation_data (ark_nlp dataset): 训练的batch文本
            evaluate_batch_size (int, optional): 验证阶段batch大小, 默认值为16
            **kwargs (optional): 其他可选参数
        """  # noqa: ignore flake8"

        self.evaluate_logs = defaultdict(int)

        kwargs = self.remove_invalid_arguments(kwargs)
        kwargs['evaluate_batch_size'] = 1

        evaluate_generator = self._on_evaluate_begin(validation_data, **kwargs)

        kwargs['epoch_step_num'] = len(evaluate_generator)

        with torch.no_grad():

            self._on_evaluate_epoch_begin(**kwargs)

            for step, inputs in enumerate(evaluate_generator):

                inputs = self._get_module_inputs_on_evaluate(inputs, **kwargs)

                # forward
                outputs = self._get_module_outputs_on_evaluate(inputs, **kwargs)

                self._on_evaluate_step_end(inputs, outputs, **kwargs)

            self._on_evaluate_epoch_end(validation_data, **kwargs)

        self._on_evaluate_end(**kwargs)

    def get_module_outputs_on_evaluate(self, inputs, **kwargs):

        with torch.no_grad():
            encoded_text = self.module.bert(inputs['input_ids'],
                                            inputs['attention_mask'])[0]
            pred_sub_heads, pred_sub_tails = self.module.get_subs(encoded_text)

        return encoded_text, pred_sub_heads, pred_sub_tails

    def on_evaluate_step_end(self, inputs, outputs, h_bar=0.5, t_bar=0.5, **kwargs):

        encoded_text, pred_sub_heads, pred_sub_tails = outputs
        token_mapping = inputs['token_mapping'][0]

        with torch.no_grad():
            sub_heads, sub_tails = np.where(pred_sub_heads.cpu()[0] > h_bar)[0], np.where(
                pred_sub_tails.cpu()[0] > t_bar)[0]

            subjects = []
            for sub_head in sub_heads:
                sub_tail = sub_tails[sub_tails >= sub_head]
                if len(sub_tail) > 0:
                    subjects.append((sub_head, sub_tail[0]))

            if subjects:
                pred_triples = set()
                repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                sub_head_mapping = torch.Tensor(len(subjects), 1,
                                                encoded_text.size(1)).zero_()
                sub_tail_mapping = torch.Tensor(len(subjects), 1,
                                                encoded_text.size(1)).zero_()
                for subject_idx, subject in enumerate(subjects):
                    sub_head_mapping[subject_idx][0][subject[0]] = 1
                    sub_tail_mapping[subject_idx][0][subject[1]] = 1
                sub_tail_mapping = sub_tail_mapping.to(repeated_encoded_text)
                sub_head_mapping = sub_head_mapping.to(repeated_encoded_text)

                pred_obj_heads, pred_obj_tails = self.module.get_objs_for_specific_sub(
                    sub_head_mapping, sub_tail_mapping, repeated_encoded_text)
                for subject_idx, sub in enumerate(subjects):

                    obj_heads, obj_tails = np.where(
                        pred_obj_heads.cpu()[subject_idx] > h_bar), np.where(
                            pred_obj_tails.cpu()[subject_idx] > t_bar)

                    for obj_head, rel_head in zip(*obj_heads):
                        for obj_tail, rel_tail in zip(*obj_tails):
                            if obj_head <= obj_tail and rel_head == rel_tail:
                                rel = self.id2cat[int(rel_head)]

                                if sub[1] - 1 >= len(token_mapping) or obj_tail - 1 >= len(token_mapping):
                                    continue

                                pred_triples.add((token_mapping[sub[0] - 1][0],
                                                  token_mapping[sub[1] - 1][-1],
                                                  rel,
                                                  token_mapping[obj_head - 1][0],
                                                  token_mapping[obj_tail - 1][-1]))

                                break
            else:
                pred_triples = set()

            gold_triples = set(to_tup(inputs['label_ids'][0]))

            if self.metric:
                self.metric.update(preds=pred_triples, labels=gold_triples)
