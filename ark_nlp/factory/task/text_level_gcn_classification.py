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

import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import tqdm
from tqdm import tqdm
import sklearn.metrics as sklearn_metrics

from ..loss_function import get_loss
from ..optimizer import get_optimizer
from ..metric import topk_accuracy
from ._task import Task
from ._sequence_classification import SequenceClassificationTask

class TextLevelGCNTask(SequenceClassificationTask):
    
    def __init__(self, *args, **kwargs):
        
        super(TextLevelGCNTask, self).__init__(*args, **kwargs)
        
    def _collate_fn(
        self, 
        batch
    ):
        batch_graph = []
        batch_input_ids = []
        batch_node_ids = []
        batch_edge_ids = []
        batch_label_ids = []
        
        for sample in batch:
            sample_graph = sample['sub_graph'].to(self.device)
            sample_graph.ndata['h'] = self.module.node_embed(torch.Tensor(sample['node_ids']).type(torch.long).to(self.device))            
            sample_graph.edata['w'] = self.module.edge_embed(torch.Tensor(sample['edge_ids']).type(torch.long).to(self.device))
            
            batch_graph.append(sample_graph)
            batch_label_ids.append(sample['label_ids'])
            
        batch_graph = dgl.batch(batch_graph)
            
        return {'sub_graph': batch_graph, 'label_ids': torch.Tensor(batch_label_ids).type(torch.long)}
        
    def _on_train_begin_record(
        self, 
        logs, 
        **kwargs
    ):
        
        logs['tr_loss'] = 0
        logs['logging_loss'] = 0
        logs['global_step'] = 0
        
        return logs
    
    def _on_backward(
        self, 
        inputs, 
        labels, 
        logits, 
        loss, 
        logs,
        verbose=True,
        gradient_accumulation_steps=1,
        grad_clip=None,
        **kwargs
    ):
                
        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            
        loss.backward() 
        
        if grad_clip != None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), grad_clip)
        
        self._on_backward_record(logs)
        
        return loss 
    
    def _on_optimize(
        self, 
        step, 
        logs, 
        gradient_accumulation_steps=1,
        **kwargs
    ):
        if (step + 1) % gradient_accumulation_steps == 0:
            self.optimizer.step()  # 更新权值
            if self.scheduler:
                self.scheduler.step()  # 更新学习率
                
            self.optimizer.zero_grad()  # 清空梯度
            
            logs['global_step'] += 1
        
        return step