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

from ark_nlp.factory.task.base._task import Task
from ark_nlp.factory.task.base._task_mixin import TaskMixin
from ark_nlp.factory.metric import SequenceClassificationMetric


class SequenceClassificationTask(TaskMixin, Task):
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
        super(SequenceClassificationTask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'SequenceLevel'
            
        if self.metric is None:
            self.metric = SequenceClassificationMetric()

    @property
    def default_optimizer(self):
        return 'adamw'

    @property
    def default_loss_function(self):
        return 'ce'
    
    def on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['loss'] += loss.item()
            
            preds = torch.argmax(logits, -1)
            
            self.metric.update(preds.detach().cpu().numpy(), inputs['label_ids'].detach().cpu().numpy())

        return None

    def on_evaluate_epoch_end(self, validation_data, evaluate_verbose=True, **kwargs):

        self.evaluate_logs.update(self.metric.result())

        if evaluate_verbose:
            
            self.log_evaluation()
                        
            # for k, v in self.evaluate_logs.items():
            #     if type(v) == float or type(v) == int:
            #         print('{} is: {:.6f}'.format(k, v))
            #     else:
            #         print('{} is: \n{}'.format(k, v))
            
            # print("********** Evaluating Done **********\n")
            # print('classification_report: \n', report)
            # print('confusion_matrix: \n', confusion_matrix)
            # print('loss is: {:.6f}'.format(self.evaluate_logs['loss'] /
            #                                self.evaluate_logs['step']))
            # print('evaluation: ', evaluate_infos)