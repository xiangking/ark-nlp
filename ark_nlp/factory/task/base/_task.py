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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset

import numpy as np

import tqdm
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data._utils.collate import default_collate

from ..loss_function import get_loss


class Task(object):
    def __init__(
        self, 
        module, 
        optimizer, 
        loss_function, 
        class_num=None,
        scheduler=None,
        n_gpu=1,
        device=None,
        cuda_device=0,
        **kwargs
    ):
        
        self.module = module
        self.optimizer = optimizer
        self.loss_function = get_loss(loss_function)
        
        self.class_num = class_num
        self.scheduler = scheduler
        
        self.n_gpu = n_gpu
        
        if device == None:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                self.device = "cpu"

        self.module.to(self.device)
        
        if self.n_gpu > 1:
            self.module = torch.nn.DataParallel(self.module)
    
    def _collate_fn(self, batch):
        return default_collate(batch)

    def _on_train_begin(self):
        pass
    
    def _on_train_begin_record(self, logs):
        pass

    def _on_epoch_begin(self,):
        pass
    
    def _on_epoch_begin_record(self, logs):
        pass

    def _on_step_begin(self):
        pass
    
    def _on_step_begin_record(self, logs):
        pass

    def _compute_loss(self,):
        pass
    
    def _compute_loss_record(self, logs):
        pass

    def _on_backward(self,):
        pass
    
    def _on_backward_record(self, logs):
        pass

    def _on_optimize(self):
        pass
    
    def _on_optimize_record(self, logs):
        pass

    def _on_step_end(self,):
        pass
    
    def _on_step_end_record(self, logs):
        pass

    def _on_epoch_end(self,):
        pass
    
    def _on_epoch_end_record(self, logs):
        pass

    def _on_train_end(self,):
        pass
    
    def _on_train_end_record(self, logs):
        pass

    def fit(self):
        pass

    def evaluate(self):
        pass

    def _get_module_inputs_on_train(self):
         pass

    def _get_module_label_on_train(self):
         pass

    def _get_module_inputs_on_eval(self):
         pass

    def _get_module_label_on_eval(self):
         pass
    