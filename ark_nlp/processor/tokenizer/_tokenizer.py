"""
# Copyright 2020 Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

import abc
import numpy as np


class BaseTokenizer(object, metaclass=abc.ABCMeta):
    """
    文本编码器基类

    Args:
        vocab: 词典类对象，用于实现文本分词和ID化
        max_seq_len (:obj:`int`): 
            预设的文本最大长度
    """  # noqa: ignore flake8"

    def __init__(self, vocab, max_seq_len):
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def tokenize(self, text):
        return self.vocab.tokenize(text)

    def pad_and_truncate(
        self,
        sequence,
        maxlen,
        dtype='int64',
        padding='post',
        truncating='post',
        value=0
    ):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x
