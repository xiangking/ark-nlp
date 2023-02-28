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
from collections import defaultdict


class Seq2SeqMetric(object):

    def __init__(self, **kwargs):
        self.preds_list = []
        self.labels_list = []

    def reset(self):
        self.preds_list = []
        self.labels_list = []

    # def lcs(self, source, target):
    #     """最长公共子序列（source和target的最长非连续子序列）
    #     返回：子序列长度, 映射关系（映射对组成的list）
    #     注意：最长公共子序列可能不止一个，所返回的映射只代表其中一个。
    #     """
    #     c = defaultdict(int)
    #     for i, si in enumerate(source, 1):
    #         for j, tj in enumerate(target, 1):
    #             if si == tj:
    #                 c[i, j] = c[i - 1, j - 1] + 1
    #             elif c[i, j - 1] > c[i - 1, j]:
    #                 c[i, j] = c[i, j - 1]
    #             else:
    #                 c[i, j] = c[i - 1, j]
    #     l, mapping = c[len(source), len(target)], []
    #     i, j = len(source) - 1, len(target) - 1
    #     while len(mapping) < l:
    #         if source[i] == target[j]:
    #             mapping.append((i, j))
    #             i, j = i - 1, j - 1
    #         elif c[i + 1, j] > c[i, j + 1]:
    #             j = j - 1
    #         else:
    #             i = i - 1
    #     return l, mapping[::-1]

    @staticmethod
    def longest_common_substring(source, target):
        """最长公共子串（source和target的最长公共切片区间）
        返回：子串长度, 所在区间（四元组）
        注意：最长公共子串可能不止一个，所返回的区间只代表其中一个。
        """
        c, l, span = defaultdict(int), 0, (0, 0, 0, 0)
        for i, si in enumerate(source, 1):
            for j, tj in enumerate(target, 1):
                if si == tj:
                    c[i, j] = c[i - 1, j - 1] + 1
                    if c[i, j] > l:
                        l = c[i, j]
                        span = (i - l, i, j - l, j)
        return l, span

    def compute(self, preds_list, labels_list, **kwargs):

        acc = 0
        f1 = 0
        for preds, labels in zip(preds_list, labels_list):
            lcs_len = Seq2SeqMetric.longest_common_substring(preds, labels)[0]
            f1 += 2. * lcs_len / (len(preds) + len(labels)) if len(preds) + len(labels) != 0 else 1
            acc += float(preds == labels)
        return acc / len(preds_list), f1 / len(preds_list)

    def result(self, **kwargs):

        acc, f1 = self.compute(self.preds_list, self.labels_list)

        return {'accuracy': acc, 'f1-score': f1}

    def update(self, preds, labels, **kwargs):

        if isinstance(preds, Tensor):
            preds = preds.numpy()

        if isinstance(labels, Tensor):
            labels = labels.numpy()

        self.preds_list.extend(preds)
        self.labels_list.extend(labels)

    @property
    def name(self):
        return ['accuracy', 'f1-score']
