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

import numpy as np
import sklearn.metrics as sklearn_metrics

from torch import Tensor


class SequenceClassificationMetric(object):
    def __init__(self):
        self.labels = []
        self.preds = []

    def reset(self):
        self.labels = []
        self.preds = []

    def compute(self, preds, labels, categories=None, **kwargs):

        if categories:
            categories = [str(category) for category in categories]

        accuracy = float(np.sum(preds == labels) / len(labels))

        f1 = float(sklearn_metrics.f1_score(labels, preds, average='macro'))

        report = sklearn_metrics.classification_report(
            labels,
            preds,
            labels=range(len(categories)) if categories else categories,
            target_names=categories)

        confusion_matrix = sklearn_metrics.confusion_matrix(labels, preds)

        return accuracy, f1, report, confusion_matrix

    def result(self, categories=None, **kwargs):

        labels = np.concatenate(self.preds)
        preds = np.concatenate(self.labels)

        accuracy, f1, report, confusion_matrix = self.compute(preds, labels, categories)

        return {
            'accuracy': accuracy,
            'f1-score': f1,
            'report': report,
            'confusion-matrix': confusion_matrix
        }

    def update(self, preds, labels, **kwargs):
        if isinstance(preds, Tensor):
            preds = preds.numpy()

        if isinstance(labels, Tensor):
            labels = labels.numpy()

        self.preds.append(preds)
        self.labels.append(labels)

    @property
    def name(self):
        return ['accuracy', 'f1-score', 'report', 'confusion-matrix']
