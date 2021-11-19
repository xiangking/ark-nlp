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

import numpy as np


class TextLevelGCNGraph(object):
    def __init__(
        self,
        graph='ngram_unweighted',
        window_size=5
    ):
        self.graph = graph
        self.window_size = window_size

    def build_graph(
        self,
        vocab,
        dataset
    ):
        if self.graph == 'ngram_unweighted':
            self.edge_num, self.edges_matrix = self.build_ngram_unweighted_graph(vocab, dataset, self.window_size)
            self.edge_weight = None
        elif self.graph == 'ngram_pmi':
            self.edge_num, self.edges_matrix, self.edge_weight = self.build_pmi_ngram_graph(vocab, dataset, self.window_size)
        else:
            raise ValueError('没有该模式的构图')

    @staticmethod
    def build_pmi_ngram_graph(
        vocab,
        dataset,
        window_size=20
    ):
        pair_count_matrix = np.zeros((vocab.vocab_size, vocab.vocab_size), dtype=int)
        word_count = np.zeros(vocab.vocab_size, dtype=int)

        for data_ in dataset.dataset:
            ids_ = vocab.convert_to_ids(vocab.tokenize(data_['text']))
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

        pmi_matrix = np.zeros((vocab.vocab_size, vocab.vocab_size), dtype=float)

        for index_ in range(vocab.vocab_size):
            for jndex_ in range(vocab.vocab_size):
                pmi_matrix[index_, jndex_] = np.log(pair_count_matrix[index_, jndex_] + 1e-8) - np.log(word_count[index_] * word_count[jndex_] + 1e-8)

        pmi_matrix = np.nan_to_num(pmi_matrix)
        pmi_matrix = np.maximum(pmi_matrix, 0.0)

        edge_weight = [0.0]
        count = 1
        adj_matrix = np.zeros((vocab.vocab_size, vocab.vocab_size), dtype=int)

        for index_ in range(vocab.vocab_size):
            for jndex_ in range(vocab.vocab_size):
                if pmi_matrix[index_, jndex_] != 0:
                    edge_weight.append(pmi_matrix[index_, jndex_])
                    adj_matrix[index_, jndex_] = count
                    count += 1

        edge_weight = np.array(edge_weight)
        edge_weight = edge_weight.reshape(-1, 1)

        edge_num = count

        return edge_num, adj_matrix, edge_weight

    @staticmethod
    def build_ngram_unweighted_graph(
        vocab,
        dataset,
        ngram=3
    ):
        count = 1
        adj_matrix = np.zeros(shape=(vocab.vocab_size, vocab.vocab_size), dtype=np.int32)

        for data_ in dataset.dataset:
            ids_ = vocab.convert_to_ids(vocab.tokenize(data_['text']))
            for src_index_, src_ in enumerate(ids_):
                for dst_index_ in range(max(0, src_index_-ngram), min(len(ids_), src_index_+ngram+1)):
                    dst_ = ids_[dst_index_]
                    if adj_matrix[src_, dst_] == 0:
                        adj_matrix[src_, dst_] = count
                        count += 1

        for token_ in range(vocab.vocab_size):
            adj_matrix[token_, token_] = count
            count += 1

        edge_num = count

        return edge_num, adj_matrix

    def get_sequence_graph(
        self,
        sequence,
        local_token2id
    ):
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
