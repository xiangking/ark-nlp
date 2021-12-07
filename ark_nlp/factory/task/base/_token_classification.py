"""
# Copyright 2020 Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

import torch

from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


class TokenClassificationTask(SequenceClassificationTask):
    """
    字符分类任务的基类
    
    Args:
        module: 深度学习模型
        optimizer: 训练模型使用的优化器名或者优化器对象
        loss_function: 训练模型使用的损失函数名或损失函数对象
        class_num (:obj:`int` or :obj:`None`, optional, defaults to None): 标签数目
        scheduler (:obj:`class`, optional, defaults to None): scheduler对象
        n_gpu (:obj:`int`, optional, defaults to 1): GPU数目
        device (:obj:`class`, optional, defaults to None): torch.device对象，当device为None时，会自动检测是否有GPU
        cuda_device (:obj:`int`, optional, defaults to 0): GPU编号，当device为None时，根据cuda_device设置device
        ema_decay (:obj:`int` or :obj:`None`, optional, defaults to None): EMA的加权系数
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):

        super(SequenceClassificationTask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'TokenLevel'

    def _compute_loss(
        self,
        inputs,
        logits,
        verbose=True,
        **kwargs
    ):

        active_loss = inputs['attention_mask'].view(-1) == 1
        active_logits = logits.view(-1, self.class_num)
        active_labels = torch.where(
            active_loss,
            inputs['label_ids'].view(-1),
            torch.tensor(self.loss_function.ignore_index).type_as(inputs['label_ids']
            )
        )

        loss = self.loss_function(active_logits, active_labels)

        return loss

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_example'] = 0

        self.evaluate_logs['labels'] = []
        self.evaluate_logs['logits'] = []
        self.evaluate_logs['input_lengths'] = []
