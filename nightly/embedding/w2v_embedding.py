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


class W2VEmbedding(object):
    def __init__(
        self, 
        sentences, 
        emb_dim, 
        window=5, 
        min_count=5, 
        workers=8, 
        sg=0, 
        iter=25, 
        hs=0
    ):
        
        self.pad_token = '<blank>'
        self.unk_token = '<unk>'
        self.emb_dim = emb_dim
        
        model = gensim.models.Word2Vec(sentences = sentences, 
                                       size = self.emb_dim, 
                                       window = window, 
                                       min_count = min_count, 
                                       workers = workers, 
                                       sg = sg, 
                                       iter = iter, 
                                       hs = hs)
        
        # 获得词典
        self.initial_tokens = self.model.dictionary
        self.initial_tokens.insert(0, self.unk_token)
        self.initial_tokens.insert(0, self.pad_token)
        
        # 获得词向量
        emb_vectors = self.model.word_vectors
        self.vocab_size = len(vocab_list)
        
        self.embeddings = np.zeros([self.vocab_size, self.embed_dim])
        for idx, trained_vec in enumerate(emb_vectors):
            self.embeddings[idx + 2] = trained_vec 

        del emb_vectors
        del model

    def add(self, token, cnt=1):
        """
        """
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx

        return idx 

    def save(self, output_vocab_path='./token2id.pkl', output_embedding_path='./word2vector.pkl'):
        with open(output_path, 'wb') as f:
            pickle.dump(self.token2id , f)
            
        with open(output_vocab_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
            
    def load(self, save_vocab_path='./token2id.pkl', save_embedding_path='./word2vector.pkl'):
        with open(save_path, 'rb') as f:
            self.token2id = pickle.load(f)
        self.id2token = self.recover_id2token()
        
        with open(save_embedding_path, 'rb') as f:
            self.embeddings = pickle.load(f)   