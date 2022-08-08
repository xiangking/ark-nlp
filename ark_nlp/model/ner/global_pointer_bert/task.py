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

from torch.utils.data._utils.collate import default_collate

from ark_nlp.factory.utils import conlleval
from ark_nlp.factory.task.base._token_classification import TokenClassificationTask


class GlobalPointerBertNERTask(TokenClassificationTask):
    """
    GlobalPointer的命名实体识别Task
    
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

    def _train_collate_fn(self, batch):

        input_ids = default_collate([f['input_ids'] for f in batch])
        attention_mask = default_collate([f['attention_mask'] for f in batch])
        token_type_ids = default_collate([f['token_type_ids'] for f in batch])
        label_ids = default_collate([f['label_ids'].to_dense() for f in batch])

        tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label_ids': label_ids,
        }
        return tensors

    def _evaluate_collate_fn(self, batch):
        return self._train_collate_fn(batch)

    def _compute_loss(
        self,
        inputs,
        logits,
        verbose=True,
        **kwargs
    ):
        loss = self.loss_function(logits, inputs['label_ids'])

        return loss

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():

            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)

            numerate, denominator = conlleval.global_pointer_f1_score(
                inputs['label_ids'].cpu(),
                logits.cpu()
            )
            self.evaluate_logs['numerate'] += numerate
            self.evaluate_logs['denominator'] += denominator

        self.evaluate_logs['example_num'] += len(inputs['label_ids'])
        self.evaluate_logs['step'] += 1
        self.evaluate_logs['loss'] += loss.item()

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        evaluate_verbose=True,
        id2cat=None,
        **kwargs
    ):

        if id2cat is None:
            id2cat = self.id2cat

        if evaluate_verbose:
            print("********** Evaluating Done **********")
            print('loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(
                self.evaluate_logs['loss'] / self.evaluate_logs['step'],
                self.evaluate_logs['numerate'],
                self.evaluate_logs['denominator'],
                2*self.evaluate_logs['numerate']/self.evaluate_logs['denominator'])
            )
