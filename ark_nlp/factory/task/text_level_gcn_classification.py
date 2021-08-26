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
import sklearn.metrics as sklearn_metrics

from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.metric import topk_accuracy
from ark_nlp.factory.task import SequenceClassificationTask


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