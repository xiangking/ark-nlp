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
from scipy import stats

from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


class UnsupervisedSimCSETask(SequenceClassificationTask):
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

        super(UnsupervisedSimCSETask, self).__init__(*args, **kwargs)

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_example'] = 0

        self.evaluate_logs['labels'] = []
        self.evaluate_logs['eval_sim'] = []

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['eval_loss'] += loss.item()

            if 'label_ids' in inputs:
                cosine_sim = self.module.cosine_sim(**inputs).cpu().numpy()
                self.evaluate_logs['eval_sim'].append(cosine_sim)
                self.evaluate_logs['labels'].append(inputs['label_ids'].cpu().numpy())

        self.evaluate_logs['eval_example'] += logits.shape[0]
        self.evaluate_logs['eval_step'] += 1

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        **kwargs
    ):

        if is_evaluate_print:
            if 'labels' in self.evaluate_logs:
                _sims = np.concatenate(self.evaluate_logs['eval_sim'], axis=0)
                _labels = np.concatenate(self.evaluate_logs['labels'], axis=0)
                spearman_corr = stats.spearmanr(_labels, _sims).correlation
                print('evaluate spearman corr is:{:.4f}, evaluate loss is:{:.6f}'.format(
                    spearman_corr,
                    self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step']
                    )
                )
            else:
                print('evaluate loss is:{:.6f}'.format(self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step']))
