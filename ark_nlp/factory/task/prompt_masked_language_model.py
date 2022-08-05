# Copyright (c) 2022 DataArk Authors. All Rights Reserved.
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
import sklearn.metrics as sklearn_metrics

from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


class PromptMLMTask(SequenceClassificationTask):
    """
        Prompt的MLM任务的Task

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
        labels = torch.squeeze(inputs['label_ids'].reshape(-1, 1))
        loss = self.loss_function(logits, labels)

        return loss

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['label'] = []
        self.evaluate_logs['logits'] = []

        return self.evaluate_logs

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['loss'] += loss.item()

            labels = inputs['label_ids'].cpu()

            logits = logits.cpu()

            batch_size = len(labels)
            vocab_size = logits.shape[1]
            label_length = labels.shape[1]

            # logits: [batch_size, label_length, vocab_size]
            logits = logits.reshape([batch_size, -1, vocab_size]).numpy()

            # [label_num, label_length]
            label_ids = np.array(
                [self.tokenizer.vocab.convert_tokens_to_ids(
                    self.tokenizer.tokenize(category)) for category in self.cat2id])

            preds = np.ones(shape=[batch_size, len(label_ids)])

            for index in range(label_length):

                preds *= logits[:, index, label_ids[:, index]]

            preds = np.argmax(preds, axis=-1)

            label_indexs = []
            for _label in labels.numpy():
                _label = "".join(
                    self.tokenizer.vocab.convert_ids_to_tokens(list(_label)))

                label_indexs.append(self.cat2id[_label])

            label_indexs = np.array(label_indexs)

        self.evaluate_logs['labels'].append(label_indexs)
        self.evaluate_logs['logits'].append(preds)

        self.evaluate_logs['accuracy'] += (label_indexs == preds).sum()

        self._on_evaluate_epoch_begin_record(inputs, outputs, logits, loss, **kwargs)

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        epoch_num=1,
        evaluate_verbose=True,
        **kwargs
    ):

        labels = np.concatenate(self.evaluate_logs['labels'], axis=0)
        preds = np.concatenate(self.evaluate_logs['logits'], axis=0)

        f1_score = sklearn_metrics.f1_score(labels, preds, average='macro')

        report = sklearn_metrics.classification_report(
            labels,
            preds,
            labels=[v for k, v in validation_data.cat2id.items()],
            target_names=[str(k) for k, v in validation_data.cat2id.items()]
        )

        confusion_matrix = sklearn_metrics.confusion_matrix(labels, preds)

        if evaluate_verbose:
            print('classification_report: \n', report)
            print('confusion_matrix_: \n', confusion_matrix)
            print('test loss is:{:.6f},test accuracy is:{:.6f},f1_score is:{:.6f}'.format(
                self.evaluate_logs['loss'] / self.evaluate_logs['step'],
                self.evaluate_logs['accuracy'] / self.evaluate_logs['example_num'],
                f1_score
                )
            )
