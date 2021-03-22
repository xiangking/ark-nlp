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
from .base_vocab import Vocab


class CharVocab(Vocab):
    
    def __init__(self, initial_tokens=None, vocab_size=None, tokenize_mode='zh'):

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

    def build_graph(self, dataset, mode='ngram_unweighted', window_size=5):
        self.window_size = window_size
        if mode == 'ngram_unweighted':
            self.edge_num, self.edges_matrix = self.build_ngram_unweighted_graph(dataset, window_size)
            self.edge_weight = None
        elif mode == 'ngram_pmi':
            self.edge_num, self.edges_matrix, self.edge_weight = self.build_pmi_ngram_graph(dataset, window_size)
        else:
            raise ValueError('没有该模式的构图')

    def build_pmi_ngram_graph(
        self, 
        dataset,
        window_size=20
    ):        
        pair_count_matrix = np.zeros((self.vocab_size, self.vocab_size), dtype=int)
        word_count = np.zeros(self.vocab_size, dtype=int)

        for data_ in dataset.dataset:
            ids_ = self.convert_to_ids(self.tokenize(data_['text']))
            for index_, token_ in enumerate(ids_):
                word_count[token_] += 1
                start_index_ = max(0, index_ - window_size)
                end_index_ = min(len(ids_), index_ + window_size)

                for jndex_ in range(start_index_, end_index_):
                    if index_ == jndex_:
                        continue
                    else:
                        target_token_ = ids_[jndex_]
                        pair_count_matrix[token_, target_token_] += 1

        total_count = np.sum(word_count)
        word_count = word_count / total_count
        pair_count_matrix = pair_count_matrix / total_count

        pmi_matrix = np.zeros((self.vocab_size, self.vocab_size), dtype=float)

        for index_ in range(self.vocab_size):
            for jndex_ in range(self.vocab_size):
                pmi_matrix[index_, jndex_] = np.log(
                    pair_count_matrix[index_, jndex_] / (word_count[index_] * word_count[jndex_]) 
                )

        pmi_matrix = np.nan_to_num(pmi_matrix)
        pmi_matrix = np.maximum(pmi_matrix, 0.0)

        edge_weight = [0.0]
        count = 1
        edge_matrix = np.zeros((self.vocab_size, self.vocab_size), dtype=int)

        for index_ in range(self.vocab_size):
            for jndex_ in range(self.vocab_size):
                if pmi_matrix[index_, jndex_] != 0:
                    edge_weight.append(pmi_matrix[index_, jndex_])
                    edge_matrix[index_, jndex_] = count
                    count += 1

        edge_weight = np.array(edge_weight)
        edge_weight = edge_weight.reshape(-1, 1)

        edge_num = count

        return edge_num, edge_matrix, edge_weight

    def build_ngram_unweighted_graph(
        self,
        dataset,
        ngram
    ):
        count = 1
        edge_matrix = np.zeros(shape=(self.vocab_size, self.vocab_size), dtype=np.int32)

        for data_ in dataset.dataset:
            ids_ = self.convert_to_ids(self.tokenize(data_['text']))
            for src_index_, src_ in enumerate(ids_):
                for dst_index_ in range(max(0, src_index_-ngram), min(len(ids_), src_index_+ngram+1)):
                    dst_ = ids_[dst_index_]
                    if edge_matrix[src_, dst_] == 0:
                        edge_matrix[src_, dst_] = count
                        count += 1

        for token_ in range(self.vocab_size):
            edge_matrix[token_, token_] = count
            count += 1

        edge_num = count

        return edge_num, edges_matrix

    def get_sequence_graph(self, sequence, local_token2id):
        local_edges = []
        edge_ids = []
        for index_, src_ in enumerate(sequence):
            local_src_ = local_token2id[src_]
            for jndex_ in range(max(0, index_ - self.window_size), min(index_ + self.window_size + 1, len(sequence))):
                dst_ = sequence[jndex_]
                local_dst_ = local_token2id[dst_]

                local_edges.append([local_src_, local_dst_])
                edge_ids.append(self.edges_matrix[src_, dst_])

            # self circle
            local_edges.append([local_src_, local_src_])
            edge_ids.append(self.edges_matrix[src_, src_])

        return local_edges, edge_ids
    
    def save(self, output_path='./token2id.pkl'):
        with open(output_path, 'wb') as f:
            pickle.dump(self.token2id , f)
            
    def load(self, save_path='./token2id.pkl'):
        with open(save_path, 'rb') as f:
            self.token2id = pickle.load(f)
        self.id2token = self.recover_id2token()