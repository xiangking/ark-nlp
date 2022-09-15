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
# Author: Chenjie Shen, jimme.shen123@gmail.com
# Status: Active


import torch

from ark_nlp.factory.metric import SpearmanCorrelationMetric
from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


class SBertTask(SequenceClassificationTask):
    """
    基于无监督的SimCSE模型文本匹配任务的Task

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

        super(SBertTask, self).__init__(*args, **kwargs)

        if 'metric' not in kwargs:
            self.metric = SpearmanCorrelationMetric()

    def on_evaluate_step_end(self, inputs, outputs, show_evaluate_loss=False, **kwargs):
        """
        计算损失和精度
        Args:
            inputs: 输入数据
            outputs: 输出数据
            show_evaluate_loss (bool, optional): 是否显示评估阶段的损失, 默认值为: False
            **kwargs: 其他参数
        """  # noqa: ignore flake8"
        with torch.no_grad():
            if show_evaluate_loss is True:
                # compute loss
                logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
                self.evaluate_logs['loss'] += loss.item()

            if 'label_ids' in inputs and show_evaluate_loss is False:

                self.metric.update(preds=self.module.cosine_sim(**inputs).cpu().numpy(),
                                   labels=inputs['label_ids'].cpu().numpy())