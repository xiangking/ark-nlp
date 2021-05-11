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

from ark_nlp.factory.utils import conlleval
from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.metric import topk_accuracy
from ark_nlp.factory.task.base._task import Task
from ark_nlp.factory.task.base._token_classification import TokenClassificationTask

from ark_nlp.factory.metric import BiaffineSpanMetrics


class BIONERTask(TokenClassificationTask):
    
    def __init__(self, *args, **kwargs):
        
        super(BIONERTask, self).__init__(*args, **kwargs)


class CRFNERTask(TokenClassificationTask):
    
    def _compute_loss(
        self, 
        inputs, 
        labels, 
        logits, 
        logs=None,
        verbose=True,
        **kwargs
    ):      
        loss = -1 * self.module.crf(emissions = logits, tags=labels, mask=inputs['attention_mask'])
        
        if logs:
            self._compute_loss_record(inputs, labels, logits, loss, logs, verbose, **kwargs)
                
        return loss

    def _on_evaluate_step_end(self, inputs, labels, logits, loss, logs, **kwargs):
        
        tags = self.module.crf.decode(logits, inputs['attention_mask'])
        tags  = tags.squeeze(0)
        
        logs['labels'].append(labels)
        logs['logits'].append(tags)
        logs['input_lengths'].append(inputs['input_lengths'])
            
        logs['nb_eval_examples'] +=  len(labels)
        logs['nb_eval_steps']  += 1
        logs['eval_loss'] += loss.item() * len(labels)
        
        return logs
        
    def _on_evaluate_end(
        self, 
        validation_data,
        logs,
        epoch=1,
        is_evaluate_print=True,
        id2cat=None,
        markup='bio',
        **kwargs):

        if id2cat == None:
            id2cat = self.id2cat
        
        self.ner_metric = conlleval.SeqEntityScore(id2cat, markup=markup)
        
        preds_ = torch.cat(logs['logits'], dim=0).cpu().numpy().tolist()        
        labels_ = torch.cat(logs['labels'], dim=0).cpu().numpy().tolist()
        input_lens_ = torch.cat(logs['input_lengths'], dim=0).cpu().numpy()
                
        for index_, label_ in enumerate(labels_):
            label_list_ = []
            pred_list_ = []
            for jndex_, _ in enumerate(label_):
                if jndex_ == 0:
                    continue
                elif jndex_ == input_lens_[index_]-1:
                    self.ner_metric.update(pred_paths=[pred_list_], label_paths=[label_list_])
                    break
                else:
                    label_list_.append(labels_[index_][jndex_])
                    pred_list_.append(preds_[index_][jndex_])        
        
        eval_info, entity_info = self.ner_metric.result()

        if is_evaluate_print:
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(logs['eval_loss'] / logs['nb_eval_steps'], 
                                                                                              eval_info['acc'], 
                                                                                              eval_info['recall'],
                                                                                              eval_info['f1']))  


class BiaffineNERTask(TokenClassificationTask):
    
    def _compute_loss(
        self, 
        inputs, 
        labels, 
        logits, 
        logs=None,
        verbose=True,
        **kwargs
    ):      
        
        span_label = labels.view(size=(-1,))
        span_logits = logits.view(size=(-1, self.class_num))
        
        span_loss = self.loss_function(span_logits, span_label)
    
        span_mask = inputs['span_mask'].view(size=(-1,))
        
        span_loss *= span_mask
        loss = torch.sum(span_loss) / inputs['span_mask'].size()[0]
        
        if logs:
            self._compute_loss_record(inputs, labels, logits, loss, logs, verbose, **kwargs)
                
        return loss

    def _compute_loss_record(
        self,
        inputs, 
        labels, 
        logits, 
        loss, 
        logs,
        verbose,
        **kwargs
    ):        
        logs['b_loss'] += loss.item() 
        logs['nb_tr_steps'] += 1
        
        return logs
        
    def _on_evaluate_step_end(self, inputs, labels, logits, loss, logs, **kwargs):
        
        logits = torch.nn.functional.softmax(logits, dim=-1)
                
        logs['labels'].append(labels.cpu())
        logs['logits'].append(logits.cpu())
            
        logs['nb_eval_examples'] +=  len(labels)
        logs['nb_eval_steps']  += 1
        logs['eval_loss'] += loss.item()
        
        return logs
        
    def _on_evaluate_end(
        self, 
        validation_data,
        logs,
        epoch=1,
        is_evaluate_print=True,
        id2cat=None,
        markup='bio',
        **kwargs):

        if id2cat == None:
            id2cat = self.id2cat
            
        biaffine_metric = BiaffineSpanMetrics()

        preds_ = torch.cat(logs['logits'], dim=0)     
        labels_ = torch.cat(logs['labels'], dim=0)

        with torch.no_grad():
            recall, precise, span_f1 = biaffine_metric(preds_, labels_)

        if is_evaluate_print:
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(logs['eval_loss'] / logs['nb_eval_steps'], 
                                                                                              precise, 
                                                                                              recall,
                                                                                              span_f1)) 