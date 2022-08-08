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

from ark_nlp.factory.utils import conlleval
from ark_nlp.factory.task.base._token_classification import TokenClassificationTask


class CrfBertNERTask(TokenClassificationTask):
    """
    +CRF命名实体模型的Task
    
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

    def _compute_loss(
        self,
        inputs,
        logits,
        verbose=True,
        **kwargs
    ):
        loss = -1 * self.module.crf(
            emissions=logits,
            tags=inputs['label_ids'].long(),
            mask=inputs['attention_mask']
        )

        return loss

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)

            tags = self.module.crf.decode(logits, inputs['attention_mask'])
            tags = tags.squeeze(0)

        self.evaluate_logs['labels'].append(inputs['label_ids'].cpu())
        self.evaluate_logs['logits'].append(tags.cpu())
        self.evaluate_logs['sequence_length'].append(inputs['sequence_length'].cpu())

        self.evaluate_logs['example_num'] += len(inputs['label_ids'])
        self.evaluate_logs['step'] += 1
        self.evaluate_logs['loss'] += loss.item()

        return logits, loss

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        evaluate_verbose=True,
        id2cat=None,
        markup='bio',
        **kwargs
    ):

        if id2cat is None:
            id2cat = self.id2cat

        self.ner_metric = conlleval.SeqEntityScore(id2cat, markup=markup)

        preds_ = torch.cat(self.evaluate_logs['logits'], dim=0).numpy().tolist()
        labels_ = torch.cat(self.evaluate_logs['labels'], dim=0).numpy().tolist()
        input_lens_ = torch.cat(self.evaluate_logs['input_lengths'], dim=0).numpy()

        for index_, label_ in enumerate(labels_):
            label_list_ = []
            pred_list_ = []
            for jndex_, _ in enumerate(label_):
                if jndex_ == 0:
                    continue
                elif jndex_ == input_lens_[index_]-1:
                    self.ner_metric.update(
                        pred_paths=[pred_list_],
                        label_paths=[label_list_]
                    )
                    break
                else:
                    label_list_.append(labels_[index_][jndex_])
                    pred_list_.append(preds_[index_][jndex_])

        evaluate_infos, entity_infos = self.ner_metric.result()

        if evaluate_verbose:
            print('evaluate loss is:{:.6f}'.format(self.evaluate_logs['loss'] /
                                                   self.evaluate_logs['step']))
            print(evaluate_infos)

            print(entity_infos)

        return None
