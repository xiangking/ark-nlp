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

class TMTask(SequenceClassificationTask):
    
    def __init__(self, *args, **kwargs):
        
        super(TMTask, self).__init__(*args, **kwargs)

    def _compute_loss_record(
        self,
        inputs, 
        logits, 
        loss, 
        verbose,
        **kwargs
    ):         
        
        if verbose:
            with torch.no_grad():
                _, preds = torch.max(logits, 1)
                self.logs['epoch_evaluation'] += torch.sum(preds == inputs['label_ids']).item() / len(inputs['label_ids'])
                
        self.logs['epoch_loss'] += loss.item() 
        self.logs['epoch_step'] += 1
        self.logs['global_step'] += 1
    
    def _on_step_end(
        self, 
        step,
        verbose=True,
        show_step=100,
        **kwargs
    ):

        if verbose and (step + 1) % show_step == 0:
            print('[{}/{}],train loss is:{:.6f},train evaluation is:{:.6f}'.format(
                step, 
                self.train_generator_lenth,
                self.logs['epoch_loss'] / self.logs['epoch_step'],
                self.logs['epoch_evaluation'] / self.logs['epoch_step']))
            
        self._on_step_end_record(**kwargs)
            
    def _on_epoch_end(
        self, 
        epoch,
        verbose=True,
        **kwargs
    ):

        if verbose:
            print('epoch:[{}],train loss is:{:.6f},train evaluation is:{:.6f} \n'.format(
                epoch,
                self.logs['epoch_loss'] / self.logs['epoch_step'],
                self.logs['epoch_evaluation'] / self.logs['epoch_step']))  