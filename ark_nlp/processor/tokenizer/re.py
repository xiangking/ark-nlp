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
from transformers.tokenization_utils import _is_control
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
    def __init__(self, max_seq_len, vocab, blank_token='[unused1]'):
        super(TransfomerTokenizer, self).__init__(max_seq_len, vocab)
        self.tokenizer_type = 'transfomer'
        self.blank_token = blank_token
        
    def tokenize(self, text):
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or _is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []

        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
            tokens.append(self.blank_token)
        
        return tokens[:-1]

    def _word_piece_tokenize(self, word):
        if word in self.vocab.vocab:
            return [word]
        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self.vocab.vocab:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop
        return tokens
    
    def sequence_to_ids(self, sequence):
        if type(sequence) == str:
            sequence = self.tokenize(sequence) 

        # 对超长序列进行截断
        if len(sequence) > self.max_seq_len - 2:
            sequence = sequence[0:(self.max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence + ['[SEP]'] 
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

        return np.array(sequence), np.array(sequence_mask), np.array(segment_ids)
    