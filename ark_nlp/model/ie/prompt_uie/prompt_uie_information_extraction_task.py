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
import torch.nn.functional as F

from ark_nlp.factory.metric import SpanMetrics
from ark_nlp.model.ie.prompt_uie.utils import get_span
from ark_nlp.model.ie.prompt_uie.utils import get_bool_ids_greater_than
from ark_nlp.factory.task.base._token_classification import TokenClassificationTask


class PromptUIETask(TokenClassificationTask):
    """
    通用信息抽取UIE的Task
    
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

    def _get_train_loss(self, inputs, outputs, **kwargs):
        loss = self._compute_loss(inputs, outputs, **kwargs)

        self._compute_loss_record(**kwargs)

        return outputs, loss

    def _get_evaluate_loss(self, inputs, outputs, **kwargs):
        loss = self._compute_loss(inputs, outputs, **kwargs)
        self._compute_loss_record(**kwargs)

        return outputs, loss

    def _compute_loss(self, inputs, logits, verbose=True, **kwargs):
        start_logits = logits[0]
        end_logits = logits[1]

        start_logits = start_logits.view(-1, 1)
        end_logits = end_logits.view(-1, 1)

        active_loss = inputs['attention_mask'].view(-1) == 1

        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]

        active_start_labels = inputs['start_label_ids'].long().view(-1, 1)[active_loss]
        active_end_labels = inputs['end_label_ids'].long().view(-1, 1)[active_loss]

        start_loss = F.binary_cross_entropy(active_start_logits,
                                            active_start_labels.to(torch.float),
                                            reduction='none')
        start_loss = torch.sum(start_loss * active_loss) / torch.sum(active_loss)

        end_loss = F.binary_cross_entropy(active_end_logits,
                                          active_end_labels.to(torch.float),
                                          reduction='none')
        end_loss = torch.sum(end_loss * active_loss) / torch.sum(active_loss)

        loss = (start_loss + end_loss) / 2.0

        return loss

    def _on_evaluate_epoch_begin(self, **kwargs):

        self.metric = SpanMetrics()

        if self.ema_decay:
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())

        self._on_epoch_begin_record(**kwargs)

    def _on_evaluate_step_end(self, inputs, logits, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, logits, **kwargs)

        S = []
        start_logits = logits[0]
        end_logits = logits[1]
        
        start_pred = start_logits.cpu().numpy().tolist()
        end_pred = end_logits.cpu().numpy().tolist()
        
        start_score_list = get_bool_ids_greater_than(start_pred)
        end_score_list = get_bool_ids_greater_than(end_pred)
        
        for index, (start_score, end_score) in enumerate(zip(start_score_list, end_score_list)):
            S = get_span(start_score, end_score)
            self.metric.update(true_subject=inputs['label_ids'][index], pred_subject=S)

        self.evaluate_logs['eval_example'] += len(inputs['label_ids'])
        self.evaluate_logs['eval_step'] += 1
        self.evaluate_logs['eval_loss'] += loss.item()

    def _on_evaluate_epoch_end(self,
                               validation_data,
                               epoch=1,
                               is_evaluate_print=True,
                               **kwargs):

        with torch.no_grad():
            eval_info = self.metric.result()

        if is_evaluate_print:
            print('eval_info: ', eval_info)

    def _train_collate_fn(self, batch):
        """将InputFeatures转换为Tensor"""

        input_ids = torch.tensor([f['input_ids'] for f in batch], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in batch],
                                      dtype=torch.long)
        token_type_ids = torch.tensor([f['token_type_ids'] for f in batch],
                                      dtype=torch.long)
        start_label_ids = torch.cat([f['start_label_ids'] for f in batch])
        end_label_ids = torch.cat([f['end_label_ids'] for f in batch])
        label_ids = [f['label_ids'] for f in batch]

        tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'start_label_ids': start_label_ids,
            'end_label_ids': end_label_ids,
            'label_ids': label_ids
        }

        return tensors

    def _evaluate_collate_fn(self, batch):
        return self._train_collate_fn(batch)
