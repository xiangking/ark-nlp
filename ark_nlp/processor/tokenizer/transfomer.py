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
import transformers 
import numpy as np

from transformers import AutoTokenizer
from torch.utils.data import Dataset
from ark_nlp.processor.tokenizer.base_tokenizer import BaseTokenizer


class TransfomerTokenizer(BaseTokenizer):
    """
    Transfomer文本编码器，用于对文本进行分词、ID化、填充等操作

    :param max_seq_len: (int) 预设的文本最大长度
    :param tokenizer: (object) 编码器，用于实现文本分词和ID化

    """  
    def __init__(self, vocab, max_seq_len):

        if isinstance(vocab, str):
            # TODO: 改成由自定义的字典所决定
            vocab = transformers.AutoTokenizer.from_pretrained(vocab)

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.tokenizer_type = 'transfomer'
    
    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')
    
    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')
    
    @staticmethod
    def recover_bert_token(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def get_token_mapping(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            ch = unicodedata.normalize('NFD', ch)
            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            token = token.lower()
            if self._is_special(token):
                token_mapping.append([]) # 如果是[CLS]或者是[SEP]之类的词，则没有对应的映射
            else:
                token = self.recover_bert_token(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

    def sequence_to_ids(self, sequence_a, sequence_b=None):
        if sequence_b is None:
            return self.sentence_to_ids(sequence_a)
        else:
            return self.pair_to_ids(sequence_a, sequence_b)

    def sentence_to_ids(self, sequence):
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

        return (np.asarray(sequence, dtype='int64'),
                np.asarray(sequence_mask, dtype='int64'), 
                np.asarray(segment_ids, dtype='int64'))
    
    def pair_to_ids(self, sequence_a, sequence_b):
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


class SentenceTokenizer(TransfomerTokenizer):
    """
    Transfomer文本编码器，用于对文本进行分词、ID化、填充等操作

    :param max_seq_len: (int) 预设的文本最大长度
    :param tokenizer: (object) 编码器，用于实现文本分词和ID化

    """  
    def sequence_to_ids(self, sequence):
    	return self.sentence_to_ids(sequence)


class PairTokenizer(TransfomerTokenizer):
    """
    Transfomer文本编码器，用于对文本进行分词、ID化、填充等操作

    :param max_seq_len: (int) 预设的文本最大长度
    :param tokenizer: (object) 编码器，用于实现文本分词和ID化

    """  
    def sequence_to_ids(self, sequence_a, sequence_b):
    	return self.pair_to_ids(sequence_a, sequence_b)


class TokensTokenizer(TransfomerTokenizer):
    """
    Transfomer文本编码器，用于对文本进行分词、ID化、填充等操作

    :param max_seq_len: (int) 预设的文本最大长度
    :param tokenizer: (object) 编码器，用于实现文本分词和ID化

    """  

    def tokenize(self, text):
        text = ' '.join([token_ for token_ in text])
        return self.vocab.tokenize(text)
    
    def sequence_to_ids(self, sequence):
        return self.sentence_to_ids(sequence)
    
    
class MRCTokenizer(TransfomerTokenizer):
    """
    Transfomer文本编码器，用于对文本进行分词、ID化、填充等操作

    :param max_seq_len: (int) 预设的文本最大长度
    :param tokenizer: (object) 编码器，用于实现文本分词和ID化

    """  

    def tokenize(self, text):
        text = ' '.join([token_ for token_ in text])
        return self.vocab.tokenize(text)
    
    def sequence_to_ids(self, sequence_a, sequence_b):
        if type(sequence_a) == str:
            sequence_a = self.tokenize(sequence_a) 

        if type(sequence_b) == str:
            sequence_b = self.tokenize(sequence_b) 
        
        # 对超长序列进行截断
        sequence_a = sequence_a
        if len(sequence_b) > ((self.max_seq_len - 3) - len(sequence_a)):
            sequence_b = sequence_b[0:(self.max_seq_len - 3) - len(sequence_a)]
            
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