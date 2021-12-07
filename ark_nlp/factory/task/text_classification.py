"""
# Copyright 2020 Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

import torch
import sklearn.metrics as sklearn_metrics

from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


class TCTask(SequenceClassificationTask):
    """
    文本分类任务的Task
    
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

        super(TCTask, self).__init__(*args, **kwargs)

    def _on_optimize_record(
        self,
        inputs,
        outputs,
        logits,
        loss,
        verbose=True,
        **kwargs
    ):
        self.logs['global_step'] += 1
        self.logs['epoch_step'] += 1

        if verbose:
            with torch.no_grad():
                _, preds = torch.max(logits, 1)
                self.logs['epoch_evaluation'] += torch.sum(preds == inputs['label_ids']).item() / len(inputs['label_ids'])

    def _on_step_end(
        self,
        step,
        inputs,
        outputs,
        logits,
        loss,
        verbose=True,
        show_step=100,
        **kwargs
    ):

        if verbose and (step + 1) % show_step == 0:
            print('[{}/{}],train loss is:{:.6f},train evaluation is:{:.6f}'.format(
                step,
                self.train_generator_lenth,
                self.logs['epoch_loss'] / self.logs['epoch_step'],
                self.logs['epoch_evaluation'] / self.logs['epoch_step']
                )
            )

        self._on_step_end_record(**kwargs)

    def _on_epoch_end(
        self,
        epoch,
        verbose=True,
        **kwargs
    ):

        if verbose:
            print('epoch:[{}],train loss is:{:.6f},train evaluation is:{:.6f} \n'.format(
                epoch,
                self.logs['epoch_loss'] / self.logs['epoch_step'],
                self.logs['epoch_evaluation'] / self.logs['epoch_step']))

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_acc'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_example'] = 0

        self.evaluate_logs['labels'] = []
        self.evaluate_logs['logits'] = []

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['eval_loss'] += loss.item()

            labels = inputs['label_ids'].cpu()
            logits = logits.cpu()

            _, preds = torch.max(logits, 1)

        self.evaluate_logs['labels'].append(labels)
        self.evaluate_logs['logits'].append(logits)

        self.evaluate_logs['eval_example'] += len(labels)
        self.evaluate_logs['eval_step'] += 1
        self.evaluate_logs['eval_acc'] += torch.sum(preds == labels.data).item()

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        **kwargs
    ):

        _labels = torch.cat(self.evaluate_logs['labels'], dim=0)
        _preds = torch.argmax(torch.cat(self.evaluate_logs['logits'], dim=0), -1)

        f1_score = sklearn_metrics.f1_score(_labels, _preds, average='macro')

        report_ = sklearn_metrics.classification_report(
            _labels,
            _preds,
            target_names=[str(_category) for _category in validation_data.categories]
        )

        confusion_matrix_ = sklearn_metrics.confusion_matrix(_labels, _preds)

        if is_evaluate_print:
            print('classification_report: \n', report_)
            print('confusion_matrix_: \n', confusion_matrix_)
            print('test loss is:{:.6f},test acc is:{:.6f},f1_score is:{:.6f}'.format(
                self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step'],
                self.evaluate_logs['eval_acc'] / self.evaluate_logs['eval_example'],
                f1_score
                )
            )