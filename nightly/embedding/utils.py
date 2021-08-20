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

import gensim
import numpy as np
from gensim.models import KeyedVectors


def l2_norm(vector):
    return (1.0 / np.linalg.norm(vector, ord=2)) * vector


def load_sub_pretrain_embedding(
    data_path, 
    vocab, 
    embed_size, 
    is_norm=True, 
    miss_token='the'
):
    model = KeyedVectors.load_word2vec_format(data_path)

    embedding_matrix = np.zeros([len(vocab.token2id), embed_size])

    for idnex_, word in enumerate(vocab.token2id):
        if idnex_ == 0:
            continue
        try:
            embedding_matrix[idnex_] = model[word]
        except KeyError:
            embedding_matrix[idnex_] = model[miss_token]
            
        if is_norm:
            embedding_matrix[idnex_] = l2_norm(embedding_matrix[idnex_])
            
    embedding_matrix = np.array(embedding_matrix)

    return embedding_matrix