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

import re
import abc
import pickle
import jieba
import unicodedata

from zhon.hanzi import punctuation
from functools import lru_cache
from collections import Counter
from transformers import BertTokenizer
from transformers.tokenization_utils import _is_control
from ark_nlp.processor.vocab._vocab import Vocab


class CharVocab(Vocab):
    
    def __init__(
        self, 
        initial_tokens=None, 
        vocab_size=None, 
        tokenize_mode='zh',
        edge_num=None, 
        adj_matrix=None, 
        edge_weight=None,
        window_size=None
    ):           
        
        self.edge_num = edge_num
        self.adj_matrix = adj_matrix
        self.edge_weight = edge_weight
        self.window_size = window_size

        self.tokenize_mode = tokenize_mode

        self.id2token = {}
        self.token2id = {}

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        
        self.vocab_size = vocab_size
        
        self.initial_tokens = self.initial_vocab(initial_tokens) if initial_tokens is not None else []
        self.vocab_size = 0
        
        self.initial_tokens.insert(0, self.unk_token)
        self.initial_tokens.insert(0, self.pad_token)
        
        for token in self.initial_tokens:
            self.add(token)
            
    def initial_vocab(self, initial_tokens):
        counter = Counter(initial_tokens)
        if self.vocab_size:
            vocab_size = self.vocab_size - 2
        else:
            vocab_size = len(counter)
        count_pairs = counter.most_common(vocab_size)
        
        tokens, _ = list(zip(*count_pairs))
        return list(tokens)
            
    def add(self, token, cnt=1):
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
            self.vocab_size += 1

        return idx
    
    def convert_to_ids(self, tokens):
        ids = [self.get_id(term) for term in tokens]
        return ids
    
    def recover_from_ids(self, ids, stop_id=None):
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens
    
    def recover_id2token(self):
        id2token_temp = {}
        for token_iter, idx_iter in self.token2id.items():
            id2token_temp[idx_iter] = token_iter
        return id2token_temp
    
    def get_id(self, token):
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]
        
    def get_token(self, idx):
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def tokenize(self, text, stop_words = None, lower=True):
        if self.tokenize_mode == 'zh':
            return CharVocab.zh_tokenize(text, stop_words, lower)
        elif self.tokenize_mode == 'en':
            return CharVocab.en_tokenize(text, stop_words, lower)
        else:
            raise ValueError('没有该分词模式')

    @classmethod
    def zh_tokenize(cls, text, stop_words = None, lower=True):
        text = re.sub(r'[%s]+' % punctuation, '', text)
        if lower:
            text = text.lower()
        tokens = [token_ for token_ in text]
        
        if stop_words:
            tokens = filter(lambda w: w not in stop_words, tokens)
        
        return list(tokens)
    
    @classmethod
    def en_tokenize(cls, text, stop_words = None, lower=True):
        if lower:
            text = text.lower()
            
        tokens = text.split()
        
        if stop_words:
            tokens = filter(lambda w: w not in stop_words, tokens)
        
        return list(tokens)
    
    def save(self, output_path='./token2id.pkl'):
        with open(output_path, 'wb') as f:
            pickle.dump(self.token2id , f)
            
    def load(self, save_path='./token2id.pkl'):
        with open(save_path, 'rb') as f:
            self.token2id = pickle.load(f)
        self.id2token = self.recover_id2token()