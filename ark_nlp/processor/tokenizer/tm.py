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

import abc
import torch
import random
import numpy as np

from torch.utils.data import Dataset
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


class TransfomerTokenizer(BaseTokenizer):
    """
    Transfomer文本编码器，用于对文本进行分词、ID化、填充等操作

    :param max_seq_len: (int) 预设的文本最大长度
    :param tokenizer: (object) 编码器，用于实现文本分词和ID化

    """  
    def __init__(self, max_seq_len, vocab):
        super(TransfomerTokenizer, self).__init__(max_seq_len, vocab)
        self.tokenizer_type = 'transfomer'
    
    def sequence_to_ids(self, sequence_a, sequence_b):
        if type(sequence_a) == str:
            sequence_a = self.tokenize(sequence_a) 

        if type(sequence_b) == str:
            sequence_b = self.tokenize(sequence_b) 
        
        # 对超长序列进行截断
        if len(sequence_a) > ((self.max_seq_len - 3)//2):
            sequence_a = sequence_a[0:(self.max_seq_len - 3)//2]
        if len(sequence_b) > ((self.max_seq_len - 3)//2):
            sequence_b = sequence_b[0:(self.max_seq_len - 3)//2]
            
        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence_a + ['[SEP]'] + sequence_b + ['[SEP]']
        segment_ids = [0] * (len(sequence_a) + 2) + [1] * (len(sequence_b) + 1)
        
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

        return (np.asarray(sequence, dtype='int64'),
                np.asarray(sequence_mask, dtype='int64'), 
                np.asarray(segment_ids, dtype='int64'))