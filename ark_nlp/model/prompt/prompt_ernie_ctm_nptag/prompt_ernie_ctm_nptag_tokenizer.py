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


class PromptErnieCtmNptagTokenizer(TransfomerTokenizer):
    """
    nptag文本编码器, 用于对文本进行分词、ID化、填充等操作

    Args:
        vocab: transformers词典类对象、词典地址或词典名, 用于实现文本分词和ID化
        max_seq_len (int): 预设的文本最大长度
    """
    def __init__(
        self,
        vocab,
        max_seq_len
    ):
        if isinstance(vocab, str):
            vocab = transformers.AutoTokenizer.from_pretrained(vocab, use_fast=False)

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.additional_special_tokens = set()
        self.tokenizer_type = 'transfomer'

        self.do_lower_case = self.vocab.do_lower_case
        self.vocab._tokenize = self._tokenize

    def sequence_to_ids(
        self,
        sequence,
        prompt,
        prompt_mode='postfix',
        return_sequence_length=False,
        **kwargs
    ):
        """
        将序列ID化

        Args:
            sequence (string or list): 输入序列
            prompt (list): 所使用的prompt, 如["是", "[MASK]"]
            prompt_mode (string):
                prompt放置在文本中的方式
                有postfix和prefix两种, postfix表示text + prompt, prefix表示prompt + text
                默认值为"postfix"
            return_sequence_length (bool, optional): 返回是否包含序列长度, 默认值为False
        """
        if type(sequence) == str:
            sequence = self.tokenize(sequence)

        if return_sequence_length:
            sequence_length = len(sequence)

        # 对超长序列进行截断
        if len(sequence) > self.max_seq_len - 1 - 2 - len(prompt):
            sequence = sequence[0:(self.max_seq_len - 1 - 2 - len(prompt))]

        # 分别在首尾拼接特殊符号
        if prompt_mode == 'postfix':
            sequence = ['[CLS0]'] + ['[CLS1]'] + sequence + prompt + ['[SEP]']
        else:
            sequence = ['[CLS0]'] + ['[CLS1]'] + prompt + sequence + ['[SEP]']

        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        segment_ids = [0] * len(sequence)

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
