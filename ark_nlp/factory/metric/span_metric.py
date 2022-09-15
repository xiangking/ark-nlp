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

from collections import Counter
from torch import Tensor


class SpanMetric(object):
    """
    从标注序列中解码出连续实体, 目前适用的模型为：[crf_bert, span_bert]
    
    """  # noqa: ignore flake8"

    def __init__(self, **kwargs):
        self.labels = []
        self.preds = []
        self.rights = []

    def reset(self):
        self.labels = []
        self.preds = []
        self.rights = []

    def compute(self, pred_num, label_num, right_num, **kwargs):

        recall = 0. if label_num == 0 else (right_num / label_num)
        precision = 0. if pred_num == 0 else (right_num / pred_num)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision +
                                                                            recall)

        return recall, precision, f1

    def result(self, categories=None, **kwargs):

        recall, precision, f1 = self.compute(len(self.preds), len(self.labels),
                                             len(self.rights))

        if categories is not None:

            id2label = {idx: label for idx, label in enumerate(categories)}

            label_counter = Counter([x[0] for x in self.labels])
            pred_counter = Counter([x[0] for x in self.preds])
            right_counter = Counter([x[0] for x in self.rights])
            target_names = []
            rows = []
            for entity_type, count in label_counter.items():

                label_num = count
                pred_num = pred_counter.get(entity_type, 0)
                right_num = right_counter.get(entity_type, 0)

                if type(entity_type) == int:
                    entity_type = id2label[entity_type]

                target_names.append(entity_type)

                type_recall, type_precision, type_f1 = self.compute(
                    pred_num, label_num, right_num)

                rows.append((entity_type, round(type_precision, 4), round(type_recall, 4),
                             round(type_f1, 4), label_num))

            # 类似 sklearn.metrics.classification_report的格式输出
            headers = ["precision", "recall", "f1-score", "support"]
            digits = 2
            longest_last_line_heading = 'weighted avg'
            name_width = max(len(cn) for cn in target_names)
            width = max(name_width, len(longest_last_line_heading), digits)
            head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
            report = head_fmt.format('', *headers, width=width)
            report += '\n\n'
            row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
            for row in rows:
                report += row_fmt.format(*row, width=width, digits=digits)
            report += '\n'

            return {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'report': report
            }
        else:
            return {'precision': precision, 'recall': recall, 'f1-score': f1}

    def update(self, preds, labels, **kwargs):
        """
        Args:
            preds (list): 
                预测span列表
                当只存在连续实体的时，格式如 [(label, start_index, end_index),...]
                当存在不连续实体的时，格式如 ['i-j-k-#-label',] 参考w2ner_bert/task/convert_index_to_text
            labels (list): 标签span列表, 格式同preds
        """  # noqa: ignore flake8"

        if isinstance(preds, Tensor):
            preds = preds.numpy()

        if isinstance(labels, Tensor):
            labels = labels.numpy()

        self.labels.extend(labels)
        self.preds.extend(preds)
        self.rights.extend(
            [pred_entity for pred_entity in preds if pred_entity in labels])

    @property
    def name(self):
        return ['precision', 'recall', 'f1-score', 'report']


class W2NERSpanMetric(SpanMetric):

    def result(self, categories=None, **kwargs):

        recall, precision, f1 = self.compute(len(self.preds), len(self.labels),
                                             len(self.rights))

        if categories is not None:
            label_counter = Counter([x.split('-')[-1] for x in self.labels])
            pred_counter = Counter([x.split('-')[-1] for x in self.preds])
            right_counter = Counter([x.split('-')[-1] for x in self.rights])
            target_names = []
            rows = []
            for entity_type, count in label_counter.items():
                label_num = count
                pred_num = pred_counter.get(entity_type, 0)
                right_num = right_counter.get(entity_type, 0)
                target_names.append(entity_type)

                recall, precision, f1 = self.compute(pred_num, label_num, right_num)

                rows.append((entity_type, round(precision,
                                                4), round(recall,
                                                          4), round(f1, 4), label_num))

            # 类似 sklearn.metrics.classification_report的格式输出
            headers = ["precision", "recall", "f1-score", "support"]
            digits = 2
            longest_last_line_heading = 'weighted avg'
            name_width = max(len(cn) for cn in target_names)
            width = max(name_width, len(longest_last_line_heading), digits)
            head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
            report = head_fmt.format('', *headers, width=width)
            report += '\n\n'
            row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
            for row in rows:
                report += row_fmt.format(*row, width=width, digits=digits)
            report += '\n'

            return {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'report': report
            }
        else:
            return {'precision': precision, 'recall': recall, 'f1-score': f1}
