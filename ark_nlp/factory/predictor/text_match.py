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


class TMPredictor(object):
    def __init__(
        self, 
        module, 
        tokernizer, 
        cat2id
    ):

        self.module = module
        self.module.task = 'SequenceLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
        self, 
        text_a,
        text_b
    ):
        input_ids = self.tokenizer.sequence_to_ids(text_a, text_b)  
        input_ids, input_mask, segment_ids = input_ids

        features = {
                'input_ids': input_ids, 
                'attention_mask': input_mask, 
                'token_type_ids': segment_ids
            }
        return features

    def _convert_to_vanilla_ids(
        self, 
        text_a, 
        text_b
    ):
        input_a_ids = vanilla_tokenizer.sequence_to_ids(row_['text_a'])
        input_b_ids = vanilla_tokenizer.sequence_to_ids(row_['text_b'])   

        features = {
                'input_a_ids': input_ids,
                'input_b_ids': input_b_ids
            }
        return features

    def _get_input_ids(
        self, 
        text_a,
        text_b
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text_a, text_b)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text_a, text_b)
        elif self.tokenizer.tokenizer_type == 'customized':
            features = self._convert_to_customized_ids(text_a, text_b)
        else:
            raise ValueError("The tokenizer type does not exist") 

    def _get_module_one_sample_inputs(
        self, 
        features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
        self, 
        text,
        topk=None,
        return_label_name=True,
        return_proba=False
    ):
        if topk == None:
            topk = len(self.cat2id) if len(self.cat2id) >2 else 1
        text_a, text_b = text
        features = self._get_input_ids(text_a, text_b)
        self.module.eval()
        
        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs)
            logit = torch.nn.functional.softmax(logit, dim=1)

        probs, indices = logit.topk(topk, dim=1, sorted=True)
        
        preds = []
        probas = []
        for pred_, proba_ in zip(indices.cpu().numpy()[0], probs.cpu().numpy()[0].tolist()):
            
            if return_label_name:
                pred_ = self.id2cat[pred_]
            
            preds.append(pred_)
                
            if return_proba:
                probas.append(proba_)
                
        if return_proba:
            return list(zip(preds, probas))

        return preds

    def _get_module_batch_inputs(
        self, 
        features
    ):
        return {col: features[col].type(torch.long).to(self.device) for col in self.inputs_cols}

    def predict_batch(
        self, 
        test_data, 
        batch_size=16, 
        shuffle=False,
        return_label_name=True,
        return_proba=False
    ):
        self.inputs_cols = test_data.dataset_cols
        
        preds = []
        probas=[]
        
        self.module.eval()
        generator = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for step, inputs in enumerate(generator):
                inputs = self._get_module_batch_inputs(inputs)

                logits = self.module(**inputs)

                preds.extend(torch.max(logits, 1)[1].cpu().numpy())  
                if return_proba:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                    probas.extend(logits.max(dim=1).values.cpu().detach().numpy())  
                
        if return_label_name:
            preds = [self.id2cat[pred_] for pred_ in preds]

        if return_proba:
            return list(zip(preds, probas))
        
        return preds