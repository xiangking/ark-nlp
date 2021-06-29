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
from torch.utils.data._utils.collate import default_collate

from ark_nlp.factory.loss_function import get_loss


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

    def _prepare_train_begin(self, **kwargs):
        pass

    def _on_train_begin(self, **kwargs):
        pass

    def _finish_train_begin(self, **kwargs):
        pass

    def _prepare_train_begin_record(self, logs, **kwargs):
        pass
    
    def _on_train_begin_record(self, logs, **kwargs):
        pass

    def _finish_train_begin_record(self, logs, **kwargs):
        pass

    def _prepare_epoch_begin(self, **kwargs):
        pass

    def _on_epoch_begin(self, **kwargs):
        pass

    def _finish_epoch_begin(self, **kwargs):
        pass

    def _prepare_epoch_begin_record(self, logs, **kwargs):
        pass
    
    def _on_epoch_begin_record(self, logs, **kwargs):
        pass

    def _finish_epoch_begin_record(self, logs, **kwargs):
        pass

    def _prepare_step_begin(self, **kwargs):
        pass

    def _on_step_begin(self, **kwargs):
        pass

    def _finish_step_begin(self, **kwargs):
        pass

    def _prepare_step_begin_record(self, logs, **kwargs):
        pass
    
    def _on_step_begin_record(self, logs, **kwargs):
        pass

    def _finish_step_begin_record(self, logs, **kwargs):
        pass

    def _prepare_compute_loss(self, **kwargs):
        pass

    def _compute_loss(self, **kwargs):
        pass

    def _finish_compute_loss(self, **kwargs):
        pass

    def _prepare_compute_loss_record(self, logs, **kwargs):
        pass
    
    def _compute_loss_record(self, logs, **kwargs):
        pass

    def _finish_compute_loss_record(self, logs, **kwargs):
        pass

    def _prepare_backward(self, **kwargs):
        pass
    
    def _on_backward(self,):
        pass

    def _finish_backward(self, **kwargs):
        pass

    def _prepare_backward_record(self, logs, **kwargs):
        pass
    
    def _on_backward_record(self, logs, **kwargs):
        pass

    def _finish_backward_record(self, logs, **kwargs):
        pass

    def _prepare_optimize(self, **kwargs):
        pass
    
    def _on_optimize(self, **kwargs):
        pass

    def _finish_optimize(self, **kwargs):
        pass

    def _prepare_optimize_record(self, logs, **kwargs):
        pass
    
    def _on_optimize_record(self, logs, **kwargs):
        pass

    def _finish_optimize_record(self, logs, **kwargs):
        pass

    def _prepare_step_end(self, **kwargs):
        pass
    
    def _on_step_end(self, **kwargs):
        pass

    def _finish_step_end(self, **kwargs):
        pass

    def _prepare_step_end_record(self, logs, **kwargs):
        pass
    
    def _on_step_end_record(self, logs, **kwargs):
        pass

    def _finish_step_end_record(self, logs, **kwargs):
        pass

    def _prepare_epoch_end(self, **kwargs):
        pass
    
    def _on_epoch_end(self, **kwargs):
        pass

    def _finish_epoch_end(self, **kwargs):
        pass

    def _prepare_epoch_end_record(self, logs, **kwargs):
        pass
    
    def _on_epoch_end_record(self, logs, **kwargs):
        pass

    def _finish_epoch_end_record(self, logs, **kwargs):
        pass

    def _prepare_train_end(self, **kwargs):
        pass
    
    def _on_train_end(self, **kwargs):
        pass

    def _finish_train_end(self, **kwargs):
        pass

    def _prepare_train_end_record(self, logs, **kwargs):
        pass
    
    def _on_train_end_record(self, logs, **kwargs):
        pass

    def _finish_train_end_record(self, logs, **kwargs):
        pass

    def _prepare_fit(self, **kwargs):
        pass
    
    def fit(self, **kwargs):
        pass

    def _finish_fit(self, **kwargs):
        pass

    def _prepare_evaluate(self, **kwargs):
        pass
    
    def evaluate(self, **kwargs):
        pass

    def _finish_evaluate(self, **kwargs):
        pass

    def _get_module_inputs_on_train(self):
         pass

    def _get_module_label_on_train(self):
         pass

    def _get_module_inputs_on_eval(self):
         pass

    def _get_module_label_on_eval(self):
         pass
    