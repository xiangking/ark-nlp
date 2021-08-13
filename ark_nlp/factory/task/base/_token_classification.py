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

from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.task.base._task import Task
from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


class TokenClassificationTask(SequenceClassificationTask):
    
    def __init__(self, *args, **kwargs):
        
        super(SequenceClassificationTask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'TokenLevel'
            
    def _compute_loss(
        self, 
        inputs, 
        logits, 
        verbose=True,
        **kwargs
    ):      
        
        active_loss = inputs['attention_mask'].view(-1) == 1
        active_logits = logits.view(-1, self.class_num)
        active_labels = torch.where(active_loss, 
                                    inputs['label_ids'].view(-1), 
                                    torch.tensor(self.loss_function.ignore_index).type_as(inputs['label_ids'])
                                   )
        loss = self.loss_function(active_logits, active_labels)
        
        self._compute_loss_record(inputs, logits, loss, verbose, **kwargs)
                
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
            
    def _on_step_end(
        self, 
        step,
        verbose=True,
        print_step=100,
        **kwargs
    ):
        if verbose and (step + 1) % print_step == 0:
            print('[{}/{}],train loss is:{:.6f}'.format(
                step, 
                self.train_generator_lenth,
                self.logs['epoch_loss'] / self.logs['epoch_step']))
                        
        self._on_step_end_record(**kwargs)
            
    def _on_epoch_end(
        self, 
        epoch,
        verbose=True,
        **kwargs
    ):
        if verbose:
            print('epoch:[{}],train loss is:{:.6f}\n'.format(
                epoch,
                self.logs['epoch_loss'] / self.logs['epoch_step']))  
            
        self._on_epoch_end_record(**kwargs)
    
    def _on_evaluate_begin_record(self, **kwargs):
        
        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_step']  = 0
        self.evaluate_logs['eval_example']  = 0
        
        self.evaluate_logs['labels'] = []
        self.evaluate_logs['logits'] = []
        self.evaluate_logs['input_lengths'] = []
                            
    def _on_evaluate_step_end(self, inputs, logits, **kwargs):

        with torch.no_grad():
            # compute loss
            loss = self._compute_loss(inputs, logits, **kwargs)
        
        self.evaluate_logs['labels'].append(inputs['label_ids'].cpu())
        self.evaluate_logs['logits'].append(logits.cpu())
        self.evaluate_logs['input_lengths'].append(inputs['input_lengths'].cpu())
            
        self.evaluate_logs['eval_example'] += len(inputs['label_ids'])
        self.evaluate_logs['eval_step']  += 1
        self.evaluate_logs['eval_loss'] += loss.item()
            
