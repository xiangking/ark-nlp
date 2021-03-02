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

from ..loss_function import get_loss
from ..optimizer import get_optimizer
from ._task import Task


class Predictor(object):
    def __init__(self, model, tokernizer, cat2id):

        self.model = model
        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def get_processed_input(self, text):
        
        input_ids = self.tokenizer.text_to_sequence(text)
        result = {
            'input_ids': input_ids
        }
        
        return result

    def predict(self, input_str=""):
        inputs = self.get_processed_input(input_str)
        
        with torch.no_grad():
            inputs = {'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device)}
            inputs['device'] = self.device
            logit = self.model.module(**inputs)
            logit = torch.nn.functional.softmax(logit, dim=1)

        probs, indices = logit.topk(len(self.cat2id), dim=1, sorted=True)
        
        res_list = []
        for pred_, prob_ in zip(indices.cpu().numpy()[0], probs.cpu().numpy()[0].tolist()):
            res_list.append([self.id2cat[pred_], prob_])
        return res_list