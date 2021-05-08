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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import tqdm
from tqdm import tqdm
import sklearn.metrics as sklearn_metrics

from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.task.base._task import Task
from ark_nlp.factory.task.base._sequence_classification import Task

from ark_nlp.factory.utils import conlleval


class TokenClassificationTask(SequenceClassificationTask):
    
    def __init__(self, *args, **kwargs):
        
        super(SequenceClassificationTask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'TokenLevel'
            
    def _on_train_begin_record(self, logs, **kwargs):
        
        logs['tr_loss'] = 0
        logs['logging_loss'] = 0
        
        return logs
            
    def _on_epoch_begin_record(self, logs):
        
        logs['b_loss'] = 0
        logs['nb_tr_steps'] = 0
        
        return logs
            
    def _compute_loss(
        self, 
        inputs, 
        labels, 
        logits, 
        logs=None,
        verbose=True,
        **kwargs
    ):      
        
        active_loss = inputs['attention_mask'].view(-1) == 1
        active_logits = logits.view(-1, self.class_num)
        active_labels = torch.where(active_loss, 
                                    labels.view(-1), 
                                    torch.tensor(self.loss_function.ignore_index).type_as(labels)
                                   )
        loss = self.loss_function(active_logits, active_labels)
        
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
        logs['b_loss'] += loss.item() * len(labels)
        logs['nb_tr_steps'] += 1
        
        return logs
    
    def _on_step_end(
        self, 
        step,
        logs,
        verbose=True,
        print_step=100,
        **kwargs
    ):
        if verbose and (step + 1) % print_step == 0:
            print('[{}/{}],train loss is:{:.6f}'.format(
                step, 
                self.train_generator_lenth,
                logs['b_loss'] / logs['nb_tr_steps']))
                        
        self._on_step_end_record(logs)
            
    def _on_epoch_end(
        self, 
        epoch,
        logs,
        verbose=True,
        **kwargs
    ):
        if verbose:
            print('epoch:[{}],train loss is:{:.6f}\n'.format(
                epoch,
                logs['b_loss'] / logs['nb_tr_steps']))  
            
        self._on_epoch_end_record(logs)
    
    def _on_evaluate_begin_record(self, logs, **kwargs):
        
        logs['eval_loss'] = 0
        logs['nb_eval_steps']  = 0
        logs['nb_eval_examples']  = 0
        
        logs['labels'] = []
        logs['logits'] = []
        logs['input_lengths'] = []
        
        return logs     
                    
    def _on_evaluate_step_end(self, inputs, labels, logits, loss, logs, **kwargs):
        
        logs['labels'].append(labels)
        logs['logits'].append(logits)
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
        preds_ = torch.argmax(torch.cat(logs['logits'], dim=0), -1).cpu().numpy().tolist()        
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