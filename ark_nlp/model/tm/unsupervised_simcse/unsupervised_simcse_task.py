"""
# Copyright 2021 Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

import torch
import numpy as np
from scipy import stats

from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


class UnsupervisedSimCSETask(SequenceClassificationTask):

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
