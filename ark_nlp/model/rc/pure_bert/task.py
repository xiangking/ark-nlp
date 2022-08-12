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
import torch.nn as nn
import sklearn.metrics as sklearn_metrics

from torch.utils.data import DataLoader
from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


class PURERCTask(SequenceClassificationTask):
    """
    基于PURE Bert的关系分类任务的Task

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
        super(PURERCTask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'TokenLevel'

    def _train_collate_fn(self, batch):
        """将InputFeatures转换为Tensor"""

        input_ids = torch.tensor([f['input_ids'] for f in batch], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in batch], dtype=torch.long)
        token_type_ids = torch.tensor([f['token_type_ids'] for f in batch], dtype=torch.long)
        position_ids = torch.tensor([f['position_ids'] for f in batch], dtype=torch.long)
        relations_idx = [f['relations_idx'] for f in batch]
        entity_pair = [f['entity_pair'] for f in batch]
        label_ids = torch.tensor([i for f in batch for i in f['label_ids']], dtype=torch.long)

        tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
            'relations_idx': relations_idx,
            'entity_pair': entity_pair,
            'label_ids': label_ids,
        }

        return tensors

    def _evaluate_collate_fn(self, batch):
        return self._train_collate_fn(batch)

    def on_optimize_record(
        self,
        inputs,
        logits,
        verbose=True,
        **kwargs
    ):
        self.logs['global_step'] += 1
        self.logs['epoch_step'] += 1

        if verbose:
            with torch.no_grad():
                _, preds = torch.max(logits, 1)
                self.logs['epoch_evaluation'] += torch.sum(preds == inputs['label_ids']).item() / len(inputs['label_ids'])

        return self.logs

    def on_step_end_record(
        self,
        step,
        verbose=True,
        show_metric_step=100,
        **kwargs
    ):

        if verbose and (step + 1) % show_metric_step == 0:
            print('[{}/{}],train loss is:{:.6f},train evaluation is:{:.6f}'.format(
                step,
                self.epoch_step_num,
                self.logs['epoch_loss'] / self.logs['epoch_step'],
                self.logs['epoch_evaluation'] / self.logs['epoch_step']
                )
            )

    def on_epoch_end_record(
        self,
        epoch,
        verbose=True,
        **kwargs
    ):
        if verbose:
            print('epoch:[{}],train loss is:{:.6f},train evaluation is:{:.6f} \n'.format(
                epoch,
                self.logs['epoch_loss'] / self.logs['epoch_step'],
                self.logs['epoch_evaluation'] / self.logs['epoch_step']))

        self.logs['epoch_loss'] = 0.0
        self.logs['epoch_step'] = 0.0

        return self.logs

    def on_evaluate_epoch_begin(self, **kwargs):

        self.evaluate_logs['labels'] = []
        self.evaluate_logs['logits'] = []

    def on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['loss'] += loss.item()

            labels = inputs['label_ids'].cpu()
            logits = logits.cpu()

            _, preds = torch.max(logits, 1)

        self.evaluate_logs['labels'].append(labels)
        self.evaluate_logs['logits'].append(logits)

        self.evaluate_logs['example_num'] += len(labels)
        self.evaluate_logs['step'] += 1
        self.evaluate_logs['accuracy'] += torch.sum(preds == labels.data).item()

        return logits, loss

    def on_evaluate_epoch_end(
        self,
        validation_data,
        evaluate_verbose=True,
        **kwargs
    ):

        labels = torch.cat(self.evaluate_logs['labels'], dim=0)
        preds = torch.argmax(torch.cat(self.evaluate_logs['logits'], dim=0), -1)

        f1_score = sklearn_metrics.f1_score(labels, preds, average='macro')

        report_ = sklearn_metrics.classification_report(
            labels,
            preds,
            target_names=[str(_category) for _category in validation_data.categories]
        )

        confusion_matrix_ = sklearn_metrics.confusion_matrix(labels, preds)

        if evaluate_verbose:
            print("********** Evaluating Done **********")
            print('classification_report: \n', report_)
            print('confusion_matrix_: \n', confusion_matrix_)
            print('loss is:{:.6f}, accuracy is:{:.6f}, f1_score is:{:.6f}'.format(
                self.evaluate_logs['loss'] / self.evaluate_logs['step'],
                self.evaluate_logs['accuracy'] / self.evaluate_logs['example_num'],
                f1_score
                )
            )
