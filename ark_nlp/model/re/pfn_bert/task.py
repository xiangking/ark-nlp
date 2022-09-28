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
import warnings
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from ark_nlp.factory.metric import RelationExtractionMetric
from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


class PFNRETask(SequenceClassificationTask):
    """
    基于PFN Bert的联合关系抽取任务的Task

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

        super(PFNRETask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'TokenLevel'

        if 'metric' not in kwargs:
            self.ner_metric = RelationExtractionMetric()
            self.re_metric = RelationExtractionMetric()

    def _train_collate_fn(self, batch):
        """将InputFeatures转换为Tensor"""

        input_ids = torch.tensor([f['input_ids'] for f in batch], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in batch],
                                      dtype=torch.long)
        ner_labels = default_collate([f['ner_labels'].to_dense() for f in batch])
        re_head_labels = default_collate([f['re_head_labels'].to_dense() for f in batch])
        re_tail_labels = default_collate([f['re_tail_labels'].to_dense() for f in batch])
        token_mapping = [f['token_mapping'] for f in batch]

        tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ner_labels': ner_labels,
            're_head_labels': re_head_labels,
            're_tail_labels': re_tail_labels,
            'token_mapping': token_mapping
        }

        return tensors

    def _evaluate_collate_fn(self, batch):
        return self._train_collate_fn(batch)

    def on_train_begin(self,
                       train_data,
                       epoch_num,
                       batch_size,
                       gradient_accumulation_step,
                       worker_num=0,
                       train_to_device_cols=None,
                       **kwargs):
        # 设置categories
        if hasattr(train_data, 'categories'):
            self.categories = train_data.categories

        # 设置 self.id2cat 和 self.cat2id
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}
            self.ner2id = train_data.ner2id

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        # 获s获取放置到GPU的变量名称列表
        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(train_data,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=worker_num,
                                     collate_fn=self._train_collate_fn)

        self.handler.epoch_step_num = len(train_generator) // gradient_accumulation_step

        self._set_optimizer(**kwargs)
        self.optimizer.zero_grad()

        self._set_scheduler(epoch_num, **kwargs)

        return train_generator

    def compute_loss(self, inputs, outputs, **kwargs):

        ner_pred, re_pred_head, re_pred_tail = outputs

        ner_label = inputs['ner_labels'].permute(1, 2, 0, 3)
        re_label_head = inputs['re_head_labels'].permute(1, 2, 0, 3)
        re_label_tail = inputs['re_tail_labels'].permute(1, 2, 0, 3)

        loss = self.loss_function(ner_pred, ner_label, re_pred_head, re_pred_tail, re_label_head, re_label_tail)

        return loss

    def get_train_loss(self, inputs, outputs, **kwargs):
        # 计算损失
        loss = self.compute_loss(inputs, outputs, **kwargs)

        return outputs, loss

    def get_evaluate_loss(self, inputs, outputs, **kwargs):
        # 计算损失
        loss = self.compute_loss(inputs, outputs, **kwargs)

        return outputs, loss

    def get_ner_index(self, tensor):
        index = (tensor == 1).nonzero(as_tuple=False)
        index_scalar = []
        for index_tup in index:
            scalar = []
            for i in index_tup:
                scalar.append(i.item())
            index_scalar.append(tuple(scalar))
        return index_scalar

    def get_re_index(self, tensor):
        index = (tensor == 1).nonzero(as_tuple=False)
        index_list = []
        for index_tup in index:
            for i in index_tup:
                index_list.append(i.item())
        return index_list

    def get_trip(self, ner_pred, re_head_pred, re_tail_pred, relation):
        seq_len = ner_pred.size(0)

        re_head_pred = re_head_pred.view(seq_len * seq_len)
        re_tail_pred = re_tail_pred.view(seq_len * seq_len)

        ner_pred = torch.sum(ner_pred, dim=-1)
        ner_pred = torch.where(ner_pred > 0, torch.ones_like(ner_pred), torch.zeros_like(ner_pred))

        ner_pred_index = self.get_ner_index(ner_pred)
        ner_map = {}  # head to [(head,tail1),(head,tail2)]
        for tup in ner_pred_index:
            if tup[0] not in ner_map:
                ner_map[tup[0]] = [tup]
            else:
                ner_map[tup[0]].append(tup)

        full_trip = []

        re_head_pred_index = self.get_re_index(re_head_pred)
        re_tail_pred_index = self.get_re_index(re_tail_pred)

        for i in range(seq_len*seq_len):
            if i in re_head_pred_index:
                subj_head = int(i // seq_len)
                obj_head = int(i % seq_len)
                if subj_head not in ner_map.keys() or obj_head not in ner_map.keys():
                    continue

                subjects = ner_map[subj_head]
                objects = ner_map[obj_head]

                for s in subjects:
                    for o in objects:
                        posit = s[1] * seq_len + o[1]
                        if posit in re_tail_pred_index:
                            full_trip.append([s, relation, o])

        return full_trip

    def _count_num(self, ner_pred, ner_label, re_pred_head, re_pred_tail, re_label_head, re_label_tail):
        ner_pred = torch.where(ner_pred>=0.5, torch.ones_like(ner_pred),
                                    torch.zeros_like(ner_pred))
        re_pred_head = torch.where(re_pred_head>=0.5, torch.ones_like(re_pred_head),
                                    torch.zeros_like(re_pred_head))
        re_pred_tail = torch.where(re_pred_tail>=0.5, torch.ones_like(re_pred_tail),
                                    torch.zeros_like(re_pred_tail))
        triple_num_list = []

        batch = ner_pred.size(2)
        for r in range(len(self.cat2id)):
            pred_num, gold_num, right_num = 0, 0, 0
            for i in range(batch):
                ner_pred_batch = ner_pred[:, :, i, :]
                ner_label_batch = ner_label[:, :, i, :]

                re_label_head_batch = re_label_head[:,:,i,r]
                re_label_tail_batch = re_label_tail[:,:,i,r]
                re_label_set = self.get_trip(ner_label_batch, re_label_head_batch, re_label_tail_batch, r)

                re_pred_head_batch = re_pred_head[:,:,i,r]
                re_pred_tail_batch = re_pred_tail[:,:,i,r]
                re_pred_set = self.get_trip(ner_pred_batch, re_pred_head_batch, re_pred_tail_batch, r)

                pred_num += len(re_pred_set)
                gold_num += len(re_label_set)

                re_right = [trip for trip in re_pred_set if trip in re_label_set]

                ner_right_batch = ner_pred_batch * ner_label_batch
                ner_right_batch = torch.sum(ner_right_batch, dim=-1)

                for trip in re_right:
                    subject = trip[0]
                    object = trip[2]
                    if ner_right_batch[subject[0], subject[1]] > 0 and ner_right_batch[object[0], object[1]] > 0:
                        right_num += 1

            triple_num_list += [pred_num, gold_num, right_num]

        return triple_num_list

    def _count_ner_num(self, ner_pred, ner_label):
        ner_pred = torch.where(ner_pred>=0.5, torch.ones_like(ner_pred),
                                    torch.zeros_like(ner_pred))
        entity_num_list = []
        for i in range(len(self.ner2id)):
            ner_pred_single = ner_pred[:, :, :, i]
            ner_label_single = ner_label[:, :, :, i]

            ner_pred_num = ner_pred_single.sum().item()
            ner_gold_num = ner_label_single.sum().item()

            ner_right = ner_pred_single * ner_label_single
            ner_right_num = ner_right.sum().item()
            entity_num_list += [ner_pred_num, ner_gold_num, ner_right_num]

        return entity_num_list

    def on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['loss'] += loss.item()

            if self.metric:
                ner_pred, re_pred_head, re_pred_tail = logits
                ner_label = inputs['ner_labels'].permute(1, 2, 0, 3)
                re_label_head = inputs['re_head_labels'].permute(1, 2, 0, 3)
                re_label_tail = inputs['re_tail_labels'].permute(1, 2, 0, 3)
                entity_num = self._count_ner_num(ner_pred, ner_label)
                triple_num = self._count_num(ner_pred, ner_label, re_pred_head, re_pred_tail, re_label_head, re_label_tail)

                for i in range(0, len(entity_num), 3):
                    self.ner_metric.update(entity_num[i], entity_num[i + 1], entity_num[i + 2])
                for i in range(0, len(triple_num), 3):
                    self.re_metric.update(triple_num[i], triple_num[i + 1], triple_num[i + 2])

        return None

    def on_evaluate_epoch_end(self, epoch_step_num, evaluate_verbose=True, **kwargs):

        if 'loss' in self.evaluate_logs:
            self.evaluate_logs['loss'] = self.evaluate_logs['loss'] / epoch_step_num

        if self.metric:
            self.evaluate_logs.update(self.ner_metric.result(categories=self.categories))

        if evaluate_verbose:
            self.log_evaluation()

        if self.metric:
            self.evaluate_logs.update(self.re_metric.result(categories=self.categories))

        if evaluate_verbose:
            self.log_evaluation()