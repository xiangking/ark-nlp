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

import numpy as np

from scipy import stats
from torch import Tensor


class SpearmanCorrelationMetric(object):
    """
    
    """  # noqa: ignore flake8"

    def __init__(self):
        self.labels = []
        self.preds = []

    def reset(self):
        self.labels = []
        self.preds = []

    def compute(self, preds, labels, **kwargs):
                
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
                
        return stats.spearmanr(labels, preds).correlation

    def result(self, **kwargs):

        spearman_correlation = self.compute(self.preds, self.labels)
        
        return {'spearman-correlation': spearman_correlation}

    def update(self, preds, labels, **kwargs):
        if isinstance(preds, Tensor):
            preds = preds.numpy()

        if isinstance(labels, Tensor):
            labels = labels.numpy()

        self.preds.append(preds)
        self.labels.append(labels)

    @property
    def name(self):
        return ['spearman-correlation']
