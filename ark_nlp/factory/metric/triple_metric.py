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


class TripleMetric(object):
    """
    
    """

  # noqa: ignore flake8"

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
        f1_score = 0. if recall + precision == 0 else (2 * precision *
                                                       recall) / (precision + recall)

        return recall, precision, f1_score

    def result(self, categories=None, **kwargs):

        recall, precision, f1_score = self.compute(self.pred_num, self.label_num,
                                                   self.right_num)

        return {'precision': precision, 'recall': recall, 'f1-score': f1_score}

    def update(self, preds, labels, **kwargs):
        """
        """  # noqa: ignore flake8"

        self.label_num += len(labels)
        self.pred_num += len(preds)
        self.right_num += len([pred for pred in preds if pred in labels])

    @property
    def name(self):
        return ['precision', 'recall', 'f1-score']
