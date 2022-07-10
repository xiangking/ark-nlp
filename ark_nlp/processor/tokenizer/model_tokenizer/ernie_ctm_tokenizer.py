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


import transformers
import numpy as np

from ark_nlp.processor.tokenizer.transfomer import TransfomerTokenizer


class ErnieCtmTokenizer(TransfomerTokenizer):
    """
    ErnieCtm模型的文本编码器，用于对文本进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名，用于实现文本分词和ID化
        max_seq_len (int): 预设的文本最大长度
        cls_num (int): CLS类特殊字符的数量，例如"[CLS0]"、"[CLS1]", 默认值为2
    """  # noqa: ignore flake8"

    def __init__(
        self,
        vocab,
        max_seq_len,
        cls_num=2
    ):

        if isinstance(vocab, str):
            # TODO: 改成由自定义的字典所决定
            vocab = transformers.AutoTokenizer.from_pretrained(
                vocab,
                use_fast=False
            )

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.additional_special_tokens = set()
        self.tokenizer_type = 'transfomer'

        self.cls_num = cls_num
        self.cls_list = ['[CLS{}]'.format(index) for index in range(self.cls_num)]

        self.do_lower_case = self.vocab.do_lower_case
        self.vocab._tokenize = self._tokenize

    def sentence_to_ids(
        self,
        sequence,
        return_sequence_length=False
    ):
        if type(sequence) == str:
            sequence = self.tokenize(sequence)

        if return_sequence_length:
            sequence_length = len(sequence)

        # 对超长序列进行截断
        if len(sequence) > self.max_seq_len - 1 - self.cls_num:
            sequence = sequence[0:(self.max_seq_len - 1 - self.cls_num)]
        # 分别在首尾拼接特殊符号
        sequence = self.cls_list + sequence + ['[SEP]']
        segment_ids = [0] * len(sequence)
        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')

        if return_sequence_length:
            return (sequence, sequence_mask, segment_ids, sequence_length)

        return (sequence, sequence_mask, segment_ids)

    def pair_to_ids(
        self,
        sequence_a,
        sequence_b,
        return_sequence_length=False,
        truncation_method='average'
    ):
        if type(sequence_a) == str:
            sequence_a = self.tokenize(sequence_a)

        if type(sequence_b) == str:
            sequence_b = self.tokenize(sequence_b)

        if return_sequence_length:
            sequence_length = (len(sequence_a), len(sequence_b))

        # 对超长序列进行截断
        if truncation_method == 'average':
            if len(sequence_a) > ((self.max_seq_len - 2 - self.cls_num)//2):
                sequence_a = sequence_a[0:(self.max_seq_len - 2 - self.cls_num)//2]
            if len(sequence_b) > ((self.max_seq_len - 2 - self.cls_num)//2):
                sequence_b = sequence_b[0:(self.max_seq_len - 2 - self.cls_num)//2]
        elif truncation_method == 'last':
            if len(sequence_b) > (self.max_seq_len - 2 - self.cls_num - len(sequence_a)):
                sequence_b = sequence_b[0:(self.max_seq_len - 2 - self.cls_num - len(sequence_a))]
        elif truncation_method == 'first':
            if len(sequence_a) > (self.max_seq_len - 2 - self.cls_num - len(sequence_b)):
                sequence_a = sequence_a[0:(self.max_seq_len - 2 - self.cls_num - len(sequence_b))]
        else:
            raise ValueError("The truncation method does not exist")

        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence_a + ['[SEP]'] + sequence_b + ['[SEP]']
        segment_ids = [0] * (len(sequence_a) + 1 + self.cls_num) + [1] * (len(sequence_b) + 1)

        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')

        if return_sequence_length:
            return (sequence, sequence_mask, segment_ids, sequence_length)

        return (sequence, sequence_mask, segment_ids)

    def _tokenize(self, text, **kwargs):
        orig_tokens = list(text)
        output_tokens = []
        for token in orig_tokens:
            if self.do_lower_case is True:
                token = token.lower()
            output_tokens.append(token)
        return output_tokens
