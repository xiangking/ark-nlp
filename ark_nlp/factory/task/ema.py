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

from ark_nlp.factory.utils.ema import EMA
from ark_nlp.factory.task import SequenceClassificationTask
from ark_nlp.factory.task import TokenClassificationTask


class EMATCTask(SequenceClassificationTask):
    def __init__(self, *args, **kwargs):
        super(EMATCTask, self).__init__(*args, **kwargs)
        self.ema = EMA(self.module.parameters(), decay=0.995)
    
    def _on_optimize(self, step, logs, **kwargs):

        if (step + 1) % gradient_accumulation_steps == 0:
            self.optimizer.step()  # 更新权值
            self.ema.update(self.module.parameters())

            if self.scheduler:
                self.scheduler.step()  # 更新学习率
                
            self.optimizer.zero_grad()  # 清空梯度
            
            logs['global_step'] += 1
        
        return step
    
    def evaluate(
        self, 
        validation_data, 
        evaluate_batch_size=16, 
        return_pred=False, 
        **kwargs
    ):
        logs = dict()
        
        generator = self._on_evaluate_begin(validation_data, evaluate_batch_size, logs, shuffle=False, **kwargs)
                
        with torch.no_grad():
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())
                        
            for step, inputs in enumerate(generator):
                
                labels = self._get_module_label_on_eval(inputs, **kwargs)
                inputs = self._get_module_inputs_on_eval(inputs, labels, **kwargs)
                
                # forward
                logits = self.module(**inputs)
                
                # compute loss
                loss = self._compute_loss(inputs, labels, logits, **kwargs)
                
                self._on_evaluate_step_end(inputs, labels, logits, loss, logs, **kwargs)
                
            self.ema.restore(self.module.parameters())
                
        self._on_evaluate_end(validation_data, logs)


class EMANERTask(TokenClassificationTask):
    def __init__(self, *args, **kwargs):
        super(EMANERTask, self).__init__(*args, **kwargs)
        self.ema = EMA(self.module.parameters(), decay=0.995)
    
    def _on_optimize(self, step, logs, **kwargs):

        if (step + 1) % gradient_accumulation_steps == 0:
            self.optimizer.step()  # 更新权值
            self.ema.update(self.module.parameters())

            if self.scheduler:
                self.scheduler.step()  # 更新学习率
                
            self.optimizer.zero_grad()  # 清空梯度
            
            logs['global_step'] += 1
        
        return step
    
    def evaluate(
        self, 
        validation_data, 
        evaluate_batch_size=16, 
        return_pred=False, 
        **kwargs
    ):
        logs = dict()
        
        generator = self._on_evaluate_begin(validation_data, evaluate_batch_size, logs, shuffle=False, **kwargs)
                
        with torch.no_grad():
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())
                        
            for step, inputs in enumerate(generator):
                
                labels = self._get_module_label_on_eval(inputs, **kwargs)
                inputs = self._get_module_inputs_on_eval(inputs, labels, **kwargs)
                
                # forward
                logits = self.module(**inputs)
                
                # compute loss
                loss = self._compute_loss(inputs, labels, logits, **kwargs)
                
                self._on_evaluate_step_end(inputs, labels, logits, loss, logs, **kwargs)
                
            self.ema.restore(self.module.parameters())
                
        self._on_evaluate_end(validation_data, logs)