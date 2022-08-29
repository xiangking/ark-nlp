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

from ark_nlp.factory.metric import BiaffineSpanMetric
from ark_nlp.factory.task.base._token_classification import TokenClassificationTask


class BiaffineBertNERTask(TokenClassificationTask):
    """
    Biaffine的命名实体识别Task
    
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

        super(BiaffineBertNERTask, self).__init__(*args, **kwargs)

        if 'metric' not in kwargs:
            self.metric = BiaffineSpanMetric()

    def compute_loss(self, inputs, logits, **kwargs):

        labels = inputs['label_ids'].view(size=(-1, ))

        loss = self.loss_function(logits.view(size=(-1, self.class_num)), labels.long())

        loss = loss * inputs['span_mask'].view(size=(-1, ))

        loss = torch.sum(loss) / inputs['span_mask'].size()[0]

        return loss

    def on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            # preds = torch.nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            self.evaluate_logs['loss'] += loss.item()

            batch_size, seq_len, hidden = inputs['label_ids'].shape
            preds = preds.view(batch_size, seq_len, hidden)

            preds = preds.view(size=(-1, ))
            labels = inputs['label_ids'].view(size=(-1, ))

            if self.metric:
                self.metric.update(preds=preds.cpu(), labels=labels.cpu())

        return logits, loss
