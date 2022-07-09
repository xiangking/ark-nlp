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

    def __init__(self, *args, tokenizer=None, **kwargs):
        super(PromptMLMTask, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer

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

            batch_size = len(labels)
            vocab_size = logits.shape[1]
            label_length = labels.shape[1]

            # logits: [batch_size, label_lenght, vocab_size]
            logits = logits.reshape([batch_size, -1, vocab_size]).numpy()

            # [label_num, label_length]
            labels_ids = np.array(
                [self.tokenizer.vocab.convert_tokens_to_ids(
                    self.tokenizer.tokenize(_cat)) for _cat in self.cat2id])

            preds = np.ones(shape=[batch_size, len(labels_ids)])

            for index in range(label_length):

                preds *= logits[:, index, labels_ids[:, index]]

            preds = np.argmax(preds, axis=-1)

            label_indexs = []
            for _label in labels.numpy():
                _label = "".join(
                    self.tokenizer.vocab.convert_ids_to_tokens(list(_label)))

                label_indexs.append(self.cat2id[_label])

            label_indexs = np.array(label_indexs)

        self.evaluate_logs['labels'].append(label_indexs)
        self.evaluate_logs['logits'].append(preds)

        self.evaluate_logs['eval_example'] += len(label_indexs)
        self.evaluate_logs['eval_step'] += 1
        self.evaluate_logs['eval_acc'] += (label_indexs == preds).sum()

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        **kwargs
    ):

        _labels = np.concatenate(self.evaluate_logs['labels'], axis=0)
        _preds = np.concatenate(self.evaluate_logs['logits'], axis=0)

        f1_score = sklearn_metrics.f1_score(_labels, _preds, average='macro')

        report_ = sklearn_metrics.classification_report(
            _labels,
            _preds,
            labels=[_v for _k, _v in validation_data.cat2id.items()],
            target_names=[str(_k) for _k, _v in validation_data.cat2id.items()]
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
