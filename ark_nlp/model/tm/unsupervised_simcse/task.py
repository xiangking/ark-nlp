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

    def _on_evaluate_epoch_begin(self, **kwargs):

        self.evaluate_logs['labels'] = []
        self.evaluate_logs['similarity'] = []

        return self.evaluate_logs

    def _on_evaluate_step_end(
        self,
        inputs,
        outputs,
        show_evaluate_loss=False,
        **kwargs
    ):
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
                cosine_sim = self.module.cosine_sim(**inputs).cpu().numpy()
                self.evaluate_logs['similarity'].append(cosine_sim)
                self.evaluate_logs['labels'].append(inputs['label_ids'].cpu().numpy())

        self.evaluate_logs['example_num'] += inputs['input_ids_a'].shape[0]
        self.evaluate_logs['step'] += 1

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        evaluate_verbose=True,
        show_evaluate_loss=False,
        **kwargs
    ):

        if evaluate_verbose:
            print("********** Evaluating Done **********\n")
            if 'labels' in self.evaluate_logs and show_evaluate_loss is False:
                sims = np.concatenate(self.evaluate_logs['similarity'], axis=0)
                labels = np.concatenate(self.evaluate_logs['labels'], axis=0)
                spearman_corr = stats.spearmanr(labels, sims).correlation
                print('spearman corr is: {:.4f}'.format(spearman_corr))

            if show_evaluate_loss is True:
                print('loss is: {:.6f}'.format(self.evaluate_logs['loss'] / self.evaluate_logs['step']))
