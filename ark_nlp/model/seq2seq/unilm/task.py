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
import torch.nn.functional as F

from ark_nlp.factory.task.base._task import Task
from ark_nlp.factory.task.base._task_mixin import TaskMixin
from ark_nlp.factory.metric.seq2seq_metric import SeqSeqMetric
# from ark_nlp.model.seq2seq.unilm.utils import longest_common_subsequence as lcs
from ark_nlp.model.seq2seq.unilm.utils import Trie, AutoRegressiveDecoder
from ark_nlp.model.seq2seq.unilm.utils import AutoQA
from torch.utils.data._utils.collate import default_collate


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """
    Numpy函数，将序列padding到同一长度

    Reference:
        [1] https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py
    """

    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


class UniLMTask(TaskMixin, Task):
    """
    序列分类任务的基类

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
        super(UniLMTask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'SequenceLevel'

        if 'metric' not in kwargs:
            self.metric = SeqSeqMetric()

        self.KG = None

        self.autoqa = AutoQA(start_id=None, end_id=self.tokenizer.vocab.sep_token_id,
                             maxlen=(self.tokenizer.max_seq_len-3)//2, device=self.device)

    @property
    def default_optimizer(self):
        return 'adamw'

    @property
    def default_loss_function(self):
        return 'ce'

    def _train_collate_fn(self, batch):
        return self.gplinker_collate_fn(batch)

    def _evaluate_collate_fn(self, batch):
        return self.gplinker_collate_fn(batch)

    def gplinker_collate_fn(self, batch):
        input_ids = default_collate([f['input_ids'] for f in batch])
        attention_mask = default_collate([f['attention_mask'] for f in batch])
        token_type_ids = default_collate([f['token_type_ids'] for f in batch])
        text_ids = [f['text_ids'] for f in batch]
        text_token_type_ids = [f['text_token_type_ids'] for f in batch]
        label = [f['label'] for f in batch]

        tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'text_ids': text_ids,
            'text_token_type_ids': text_token_type_ids,
            'label': label,
        }
        return tensors

    def get_train_loss(self, inputs, outputs, **kwargs):
        # 计算损失
        loss = self.compute_loss(inputs, outputs, **kwargs)

        return outputs, loss

    def get_evaluate_loss(self, inputs, outputs, **kwargs):
        # 计算损失
        loss = self.compute_loss(inputs, outputs, **kwargs)

        return outputs, loss

    # def loss_function(self, y_true, y_pred, mask_zero=False):
    #     zeros = torch.zeros_like(y_pred[..., :1])
    #     y_pred = torch.cat([y_pred, zeros], dim=-1)
    #     if mask_zero:
    #         infs = zeros + 1e12
    #         y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
    #
    #     y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    #     y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    #     if mask_zero:
    #         y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
    #         y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    #     pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    #     all_loss = torch.logsumexp(y_pred, dim=-1)
    #     aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    #     aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
    #     neg_loss = all_loss + torch.log(aux_loss)
    #
    #     return pos_loss + neg_loss
    #
    # def compute_loss(self, inputs, outputs, **kwargs):
    #     y_true, y_mask, y_pred = inputs['input_ids'], inputs['token_type_ids'], outputs
    #     y_true = y_true[:, 1:].contiguous()  # 目标token_ids
    #     y_mask = y_mask[:, 1:].contiguous()  # segment_ids，刚好指示了要预测的部分
    #     y_pred = y_pred[:, :-1, :].contiguous()  # 预测序列，错开一位
    #     loss = self.loss_function(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1))
    #     loss = torch.sum(loss * y_mask) / torch.sum(y_mask)
    #     return loss

    def compute_loss(self, inputs, outputs, **kwargs):
        loss = self.loss_function(outputs, inputs)
        return loss

    def on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)

            self.evaluate_logs['loss'] += loss.item()

            # if self.metric:
            preds = self.autoqa.generate(inputs, self.tokenizer, self.module, self.KG, topk=1, min_ends=3)

            self.metric.update(preds, inputs['label'])

        return None