"""
# Copyright Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

from ark_nlp.processor.tokenizer._tokenizer import BaseTokenizer


class VanillaTokenizer(BaseTokenizer):
    """
    文本编码器，用于对文本进行分词、ID化、填充等操作

    :param max_seq_len: (int) 预设的文本最大长度
    :param tokenizer: (object) 编码器，用于实现文本分词和ID化

    Reference:
        [1] https://github.com/dasiki/https-github.com-ami66-ChineseTextClassifier
    """
    def __init__(self, max_seq_len, vocab):
        super(VanillaTokenizer, self).__init__(max_seq_len, vocab)
        self.tokenizer_type = 'vanilla'

    def sequence_to_ids(
        self,
        sequence,
        reverse=False,
        padding='post',
        truncating='post'
    ):
        if type(sequence) == str:
            sequence = self.tokenize(sequence) 

        sequence = self.vocab.convert_to_ids(sequence)
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]

        return self.pad_and_truncate(sequence,
                                     self.max_seq_len,
                                     padding=padding,
                                     truncating=truncating)
