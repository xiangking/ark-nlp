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
import sklearn.metrics as sklearn_metrics

from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask as TCTask


# class TCTask(SequenceClassificationTask):
#     """
#     文本分类任务的Task
    
#     Args:
#         module: 深度学习模型
#         optimizer (str or torch.optim.Optimizer or None, optional): 训练模型使用的优化器名或者优化器对象, 默认值为: None
#         loss_function (str or object or None, optional): 训练模型使用的损失函数名或损失函数对象, 默认值为: None
#         scheduler (torch.optim.lr_scheduler.LambdaLR, optional): scheduler对象, 默认值为: None
#         tokenizer (object or None, optional): 分词器, 默认值为: None
#         class_num (int or None, optional): 标签数目, 默认值为: None
#         gpu_num (int, optional): GPU数目, 默认值为: 1
#         device (torch.device, optional): torch.device对象, 当device为None时, 会自动检测是否有GPU
#         cuda_device (int, optional): GPU编号, 当device为None时, 根据cuda_device设置device, 默认值为: 0
#         ema_decay (int or None, optional): EMA的加权系数, 默认值为: None
#         **kwargs (optional): 其他可选参数
#     """  # noqa: ignore flake8"
#     pass

#     def on_evaluate_epoch_begin(self, **kwargs):

#         self.evaluate_logs['labels'] = []
#         self.evaluate_logs['logits'] = []

#         return None

#     def on_evaluate_step_end(self, inputs, outputs, **kwargs):

#         with torch.no_grad():
#             # compute loss
#             logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
#             self.evaluate_logs['loss'] += loss.item()

#             labels = inputs['label_ids'].cpu()
#             logits = logits.cpu()

#             _, preds = torch.max(logits, 1)

#         self.evaluate_logs['labels'].append(labels)
#         self.evaluate_logs['logits'].append(logits)

#         self.evaluate_logs['example_num'] += len(labels)
#         self.evaluate_logs['step'] += 1
#         self.evaluate_logs['accuracy'] += torch.sum(preds == labels.data).item()

#         return logits, loss

#     def on_evaluate_epoch_end(self, validation_data, evaluate_verbose=True, **kwargs):

#         labels = torch.cat(self.evaluate_logs['labels'], dim=0)
#         preds = torch.argmax(torch.cat(self.evaluate_logs['logits'], dim=0), -1)

#         self.evaluate_logs['f1_score'] = sklearn_metrics.f1_score(labels, preds, average='macro')
#         self.evaluate_logs['accuracy'] = self.evaluate_logs['accuracy'] / self.evaluate_logs['example_num']

#         report = sklearn_metrics.classification_report(
#             labels,
#             preds,
#             target_names=[str(category) for category in validation_data.categories])

#         confusion_matrix = sklearn_metrics.confusion_matrix(labels, preds)
        
#         if evaluate_verbose:
#             print("********** Evaluating Done **********\n")
#             print('classification_report: \n', report)
#             print('confusion_matrix: \n', confusion_matrix)
#             print('loss is: {:.6f}, accuracy is: {:.6f}, f1_score is: {:.6f}'.format(
#                 self.evaluate_logs['loss'] / self.evaluate_logs['step'],
#                 self.evaluate_logs['accuracy'],
#                 self.evaluate_logs['f1_score']))
