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

import math
import random
import numpy as np
import pandas as pd

from collections import defaultdict


def convert_ner_task_uie_df(df, negative_ratio=5):

    negative_examples = []
    positive_examples = []

    label_type_set = set()

    for labels in df['label']:
        for label in labels:
            label_type_set.add(label['type'])

    for text, labels in zip(df['text'], df['label']):
        type2entities = defaultdict(list)

        for label in labels:
            type2entities[label['type']].append(label)

        positive_num = len(type2entities)

        for type_name, entities in type2entities.items():
            positive_examples.append({
                'text': text,
                'label': entities,
                'condition': type_name
            })

        if negative_ratio == 0:
            continue

        redundant_label_type_list = list(label_type_set - set(type2entities.keys()))
        redundant_label_type_list.sort()

        # 负样本抽样
        if positive_num != 0:
            actual_ratio = math.ceil(len(redundant_label_type_list) / positive_num)
        else:
            positive_num, actual_ratio = 1, 0

        if actual_ratio <= negative_ratio or negative_ratio == -1:
            idxs = [k for k in range(len(redundant_label_type_list))]
        else:
            idxs = random.sample(range(0, len(redundant_label_type_list)),
                                 negative_ratio * positive_num)

        for idx in idxs:
            negative_examples.append({
                'text': text,
                'label': [],
                'condition': redundant_label_type_list[idx]
            })

    return pd.DataFrame(positive_examples + negative_examples)


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    根据概率列表输出span的index列表

    Args:
        probs (list):
            概率列表: [[每个位置是span的首index的概率], [每个位置是span的尾index的概率]]
            例如: [[0.1, 0.1, 0.2, 0.5, 0.1, 0.3], [0.7, 0.6, 0.1, 0.1, 0.1, 0.1]]
        limit (float, optional): 阈值, 默认值为0.5
        return_prob (bool, optional): 返回是否带上概率, 默认值为False

    Returns:
        list: span的index列表, 例如: [[3], [0, 1]]

    Reference:
        [1] https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/utils.py
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def get_span(start_ids, end_ids, with_prob=False):
    """
    根据span的首index列表和尾index列表生成span

    Args:
        start_ids (list): span的首index列表: [index], 例如: [1, 2, 10]
        end_ids (list): span的尾index列表: [index], 例如: [4, 12]
        with_prob (bool, optional): 输入的列表是否带上概率, 默认值为False

    Returns:
        set: span列表, 例如: set((2, 4), (10, 12))

    Reference:
        [1] https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/utils.py
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            if start_ids[start_pointer][0] == end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer][0] < end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer][0] > end_ids[end_pointer][0]:
                end_pointer += 1
                continue
        else:
            if start_ids[start_pointer] == end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer] < end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer] > end_ids[end_pointer]:
                end_pointer += 1
                continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)

    return result
