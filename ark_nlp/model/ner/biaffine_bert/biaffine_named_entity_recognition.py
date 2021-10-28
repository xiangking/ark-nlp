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

from ark_nlp.factory.metric import BiaffineSpanMetrics
from ark_nlp.factory.task.base._token_classification import TokenClassificationTask


class BiaffineNERTask(TokenClassificationTask):

    def _compute_loss(
        self,
        inputs,
        logits,
        verbose=True,
        **kwargs
    ):

        span_label = inputs['label_ids'].view(size=(-1,))
        span_logits = logits.view(size=(-1, self.class_num))

        span_loss = self.loss_function(span_logits, span_label)

        span_mask = inputs['span_mask'].view(size=(-1,))

        span_loss *= span_mask
        loss = torch.sum(span_loss) / inputs['span_mask'].size()[0]

        return loss

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            logits = torch.nn.functional.softmax(logits, dim=-1)
            self.evaluate_logs['eval_loss'] += loss.item()

        self.evaluate_logs['labels'].append(inputs['label_ids'].cpu())
        self.evaluate_logs['logits'].append(logits.cpu())

        self.evaluate_logs['eval_example'] += len(inputs['label_ids'])
        self.evaluate_logs['eval_step'] += 1

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        id2cat=None,
        **kwargs
    ):

        if id2cat is None:
            id2cat = self.id2cat

        biaffine_metric = BiaffineSpanMetrics()

        preds_ = torch.cat(self.evaluate_logs['logits'], dim=0)
        labels_ = torch.cat(self.evaluate_logs['labels'], dim=0)

        with torch.no_grad():
            recall, precise, span_f1 = biaffine_metric(preds_, labels_)

        if is_evaluate_print:
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(
                self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step'],
                precise,
                recall,
                span_f1)
            )
