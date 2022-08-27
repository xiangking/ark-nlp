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
# Author: Chenjie Shen, jimme.shen123@gmail.com
#         Xiang Wang, xiangking1995@163.com
# Status: Active

from torch import Tensor


class GlobalPointerMetric(object):

    def __init__(self, **kwargs):
        self.right_num = 0
        self.label_num = 0
        self.pred_num = 0

    def reset(self):
        self.right_num = 0
        self.label_num = 0
        self.pred_num = 0

    def compute(self, pred_num, label_num, right_num, **kwargs):

        recall = 0. if label_num == 0 else (right_num / label_num)
        precision = 0. if pred_num == 0 else (right_num / pred_num)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision +
                                                                            recall)

        return recall, precision, f1

    def result(self, **kwargs):

        recall, precision, f1 = self.compute(self.pred_num, self.label_num,
                                             self.right_num)

        return {'precision': precision, 'recall': recall, 'f1-score': f1}

    def update(self, preds, labels, **kwargs):

        if isinstance(preds, Tensor):
            preds = preds.numpy()

        if isinstance(labels, Tensor):
            labels = labels.numpy()

        self.right_num += float(((preds > 0) * (labels > 0)).sum())
        self.label_num += float((labels > 0).sum())
        self.pred_num += float((preds > 0).sum())

    @property
    def name(self):
        return ['precision', 'recall', 'f1-score']
