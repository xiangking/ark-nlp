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

import time
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics as sklearn_metrics

from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.task.base._task import Task
from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


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


def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }


class PRGCRETask(SequenceClassificationTask):
    
    def __init__(self, *args, **kwargs):
        
        super(PRGCRETask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'TokenLevel'
            
    def _collate_fn_train(self, features):
        """将InputFeatures转换为Tensor"""
        
        input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        seq_tags = torch.tensor([f['seq_tags'] for f in features], dtype=torch.long)
        poten_relations = torch.tensor([f['potential_rels'] for f in features], dtype=torch.long)
        corres_tags = torch.tensor([f['corres_tags'] for f in features], dtype=torch.long)
        rel_tags = torch.tensor([f['rel_tags'] for f in features], dtype=torch.long)
        token_mapping = [f['token_mapping'] for f in features]
        
        tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'seq_tags': seq_tags,
            'potential_rels': poten_relations,
            'corres_tags': corres_tags,
            'rel_tags': rel_tags,
            'token_mapping': token_mapping
        }
                
        return tensors
    
    def _collate_fn_evaluate(self, features):
        """将InputFeatures转换为Tensor"""
        
        input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        triples = [f['triples'] for f in features]
        token_mapping = [f['token_mapping'] for f in features]

        tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'triples': triples,
            'token_mapping': token_mapping
        }

        return tensors

    def _on_train_begin(
        self, 
        train_data, 
        validation_data, 
        batch_size,
        lr, 
        params, 
        shuffle,
        train_to_device_cols=None,
        **kwargs
    ):
        
        if self.class_num == None:
            self.class_num = train_data.class_num  
        
        if train_to_device_cols == None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols
                        
        train_generator = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn_train)
        self.train_generator_lenth = len(train_generator)
            
        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()
        
        self.module.train()
        
        self._on_train_begin_record(**kwargs)
        
        return train_generator
            
    def _compute_loss(
        self, 
        inputs, 
        logits, 
        verbose=True,
        **kwargs
    ):  
        batch_size, _ = inputs['input_ids'].size()
        
        output_sub, output_obj, corres_pred, rel_pred = logits
        
        mask_tmp1 = inputs['attention_mask'].unsqueeze(-1)
        mask_tmp2 = inputs['attention_mask'].unsqueeze(1)
        corres_mask = mask_tmp1 * mask_tmp2
        
        attention_mask = inputs['attention_mask'].view(-1)
        # sequence label loss
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_seq_sub = (loss_func(output_sub.view(-1, self.module.seq_tag_size),
                                  inputs['seq_tags'][:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
        loss_seq_obj = (loss_func(output_obj.view(-1, self.module.seq_tag_size),
                                  inputs['seq_tags'][:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
        loss_seq = (loss_seq_sub + loss_seq_obj) / 2
        # init
        loss_matrix, loss_rel = torch.tensor(0), torch.tensor(0)
        
        corres_pred = corres_pred.view(batch_size, -1)
        corres_mask = corres_mask.view(batch_size, -1)
        corres_tags = inputs['corres_tags'].view(batch_size, -1)
        
        loss_func = nn.BCEWithLogitsLoss(reduction='none')
        
        loss_matrix = (loss_func(corres_pred,
                                 corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

        loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        loss_rel = loss_func(rel_pred, inputs['rel_tags'].float())

        loss = loss_seq + loss_matrix + loss_rel              
            
        return loss
    
    def _compute_loss_record(
        self,
        inputs, 
        logits, 
        loss, 
        verbose,
        **kwargs
    ):         
                
        self.logs['epoch_loss'] += loss.item() 
        self.logs['epoch_step'] += 1
        self.logs['global_step'] += 1

    def fit(
        self, 
        train_data=None, 
        validation_data=None, 
        lr=False,
        params=None,
        batch_size=32,
        epochs=1,
        **kwargs
    ):
        self.logs = dict()

        self.id2cat = train_data.id2cat

        train_generator = self._on_train_begin(train_data, validation_data, batch_size, lr, params, shuffle=True, **kwargs)

        for epoch in range(epochs):

            self._on_epoch_begin(**kwargs)

            for step, inputs in enumerate(tqdm(train_generator)):

                self._on_step_begin(epoch, step, inputs, **kwargs)

                inputs = self._get_module_inputs_on_train(inputs, **kwargs)

                # forward
                logits = self.module(**inputs)

                # 计算损失
                loss = self._get_train_loss(inputs, logits, **kwargs)

                # loss backword
                loss = self._on_backward(inputs, logits, loss, **kwargs)

                # optimize
                step = self._on_optimize(step, **kwargs)

                # setp evaluate
                self._on_step_end(step, inputs, logits, loss, **kwargs)

            self._on_epoch_end(epoch, **kwargs)

            if validation_data is not None:
                self.evaluate(validation_data, **kwargs)

        self._on_train_end(**kwargs)

    def _on_evaluate_begin(
        self, 
        validation_data, 
        batch_size, 
        shuffle, 
        evaluate_to_device_cols=None,
        **kwargs
    ):

        self.evaluate_id2sublabel = validation_data.sublabel2id
        self.evaluate_id2oblabel = validation_data.oblabel2id
        
        if evaluate_to_device_cols == None:
            self.evaluate_to_device_cols = validation_data.to_device_cols
        else:
            self.evaluate_to_device_cols = evaluate_to_device_cols

        generator = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn_evaluate)

        self.module.eval()

        self._on_evaluate_begin_record(**kwargs)

        return generator

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['correct_num'] = 0
        self.evaluate_logs['predict_num'] = 0
        self.evaluate_logs['gold_num'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_example'] = 0
        
        
    def _on_evaluate_step_end(self, inputs, logits, corres_threshold=0.5, **kwargs):
        
        batch_size, _ = inputs['input_ids'].size()
        token_mappings = inputs['token_mapping']
        
        output_sub, output_obj, corres_pred, pred_rels, xi = logits
        
        pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
        pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
        pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
        
        mask_tmp1 = inputs['attention_mask'].unsqueeze(-1)
        mask_tmp2 = inputs['attention_mask'].unsqueeze(1)
        corres_mask = mask_tmp1 * mask_tmp2
        
        corres_pred = torch.sigmoid(corres_pred) * corres_mask
        pre_corres = torch.where(corres_pred > corres_threshold,
                                         torch.ones(corres_pred.size(), device=corres_pred.device),
                                         torch.zeros(corres_pred.size(), device=corres_pred.device))
        
        pred_seqs = pred_seqs.detach().cpu().numpy()
        pre_corres = pre_corres.detach().cpu().numpy()
        
        xi = np.array(xi)
        pred_rels = pred_rels.detach().cpu().numpy()
        xi_index = np.cumsum(xi).tolist()
        xi_index.insert(0, 0)
        
        for idx in range(batch_size):
            pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                             pre_corres=pre_corres[idx],
                                             pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                             label2idx_sub=self.evaluate_id2sublabel,
                                             label2idx_obj=self.evaluate_id2oblabel)

            self.evaluate_logs['correct_num'] += len(set(pre_triples) & set(inputs['triples'][idx]))
            self.evaluate_logs['predict_num'] += len(set(pre_triples))
            self.evaluate_logs['gold_num'] += len(set(inputs['triples'][idx]))
        
    def _on_evaluate_epoch_end(
        self, 
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        **kwargs
    ):
        
        metrics = get_metrics(self.evaluate_logs['correct_num'], self.evaluate_logs['predict_num'] ,  self.evaluate_logs['gold_num'])
        metrics_str = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics.items())
        
        print(metrics_str)
    
    def evaluate(
        self, 
        validation_data, 
        evaluate_batch_size=16, 
        return_pred=False, 
        **kwargs
    ):
        self.evaluate_logs = dict()
        
        generator = self._on_evaluate_begin(validation_data, evaluate_batch_size, shuffle=False, **kwargs)
                
        with torch.no_grad():

            self._on_evaluate_epoch_begin(**kwargs)

            for step, inputs in enumerate(generator):
                
                inputs = self._get_module_inputs_on_eval(inputs, **kwargs)
                
                # forward
                logits = self.module(**inputs)
                
                self._on_evaluate_step_end(inputs, logits, **kwargs)
            
            self._on_evaluate_epoch_end(validation_data, **kwargs)
                
        self._on_evaluate_end(**kwargs)