"""
# Copyright 2021 Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

import torch
import numpy as np


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    content = tag_name.split('-')
    tag_class = content[0]
    if len(content) == 1:
        return tag_class
    ht = content[-1]
    return tag_class, ht


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: np.array[4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default1 = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default1 and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default1:
            res = get_chunk_type(tok, idx_to_tag)
            if len(res) == 1:
                continue
            tok_chunk_class, ht = get_chunk_type(tok, idx_to_tag)
            tok_chunk_type = ht
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def tag_mapping_corres(predict_tags, pre_corres, pre_rels=None, label2idx_sub=None, label2idx_obj=None):
    """
    Args:
        predict_tags: np.array, (xi, 2, max_sen_len)
        pre_corres: (seq_len, seq_len)
        pre_rels: (xi,)
    """
    rel_num = predict_tags.shape[0]
    pre_triples = []
    for idx in range(rel_num):
        heads, tails = [], []
        pred_chunks_sub = get_chunks(predict_tags[idx][0], label2idx_sub)
        pred_chunks_obj = get_chunks(predict_tags[idx][1], label2idx_obj)
        pred_chunks = pred_chunks_sub + pred_chunks_obj
        for ch in pred_chunks:
            if ch[0] == 'H':
                heads.append(ch)
            elif ch[0] == 'T':
                tails.append(ch)
        retain_hts = [(h, t) for h in heads for t in tails if pre_corres[h[1]][t[1]] == 1]
        for h_t in retain_hts:
            if pre_rels is not None:
                triple = list(h_t) + [pre_rels[idx]]
            else:
                triple = list(h_t) + [idx]
            pre_triples.append(tuple(triple))
    return pre_triples


class PRGCREPredictor(object):
    """
    CasRel bert模型的联合关系抽取任务的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
        corres_threshold (:obj:`float`, optional, defaults to 0.5): global correspondence的阈值
    """  # noqa: ignore flake8"

    def __init__(
        self,
        module,
        tokernizer,
        cat2id,
        corres_threshold=0.5
    ):
        self.module = module
        self.module.task = 'TokenLevel'

        self.corres_threshold = corres_threshold

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

        self.sublabel2id = {"B-H": 1, "I-H": 2, "O": 0}
        self.oblabel2id = {"B-T": 1, "I-T": 2, "O": 0}

    def _convert_to_transfomer_ids(
        self,
        text
    ):
        if len(text) > self.tokenizer.max_seq_len - 2:
            text = text[:self.tokenizer.max_seq_len - 2]

        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens, is_mapping_index=False)
        # index_token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        # start_mapping = {j[0]: i for i, j in enumerate(index_token_mapping) if j}
        # end_mapping = {j[-1]: i for i, j in enumerate(index_token_mapping) if j}

        input_ids, input_mask, segment_ids = self.tokenizer.sequence_to_ids(tokens)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_mapping': token_mapping
        }

        return features

    def _get_input_ids(
        self,
        text
    ):
        if self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
        self,
        features
    ):
        inputs = {}
        for col in features:
            if isinstance(features[col], np.ndarray):
                inputs[col] = torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device)
            else:
                inputs[col] = features[col]

        return inputs

    def predict_one_sample(
        self,
        text='',
    ):
        features = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():

            inputs = self._get_module_one_sample_inputs(features)

            logits = self.module(**inputs)

            token_mapping = inputs['token_mapping']

            output_sub, output_obj, corres_pred, pred_rels, xi = logits

            pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
            pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
            pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)

            mask_tmp1 = inputs['attention_mask'].unsqueeze(-1)
            mask_tmp2 = inputs['attention_mask'].unsqueeze(1)
            corres_mask = mask_tmp1 * mask_tmp2

            corres_pred = torch.sigmoid(corres_pred) * corres_mask
            pre_corres = torch.where(
                corres_pred > self.corres_threshold,
                torch.ones(corres_pred.size(), device=corres_pred.device),
                torch.zeros(corres_pred.size(), device=corres_pred.device)
            )

            pred_seqs = pred_seqs.detach().cpu().numpy()
            pre_corres = pre_corres.detach().cpu().numpy()

            xi = np.array(xi)
            pred_rels = pred_rels.detach().cpu().numpy()
            xi_index = np.cumsum(xi).tolist()
            xi_index.insert(0, 0)

            pre_triples = tag_mapping_corres(
                predict_tags=pred_seqs[xi_index[0]:xi_index[1]],
                pre_corres=pre_corres[0],
                pre_rels=pred_rels[xi_index[0]:xi_index[1]],
                label2idx_sub=self.sublabel2id,
                label2idx_obj=self.oblabel2id
            )

            triple_set = set()
            for _pre_triple in pre_triples:
                sub = ''.join([token_mapping[index_] for index_ in range(_pre_triple[0][1]-1, _pre_triple[0][2]-1)])
                obj = ''.join([token_mapping[index_] for index_ in range(_pre_triple[1][1]-1, _pre_triple[1][2]-1)])
                rel = self.id2cat[_pre_triple[2]]

                triple_set.add((sub, rel, obj))

        return list(triple_set)
