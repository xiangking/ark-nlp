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
from torch.autograd import grad
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ark_nlp.factory.utils.conlleval import get_entities


class BIONERPredictor(object):
    def __init__(
        self, 
        module, 
        tokernizer, 
        cat2id,
        markup='bio'
    ):
        self.markup = markup

        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
        self, 
        text
    ):
        input_ids = self.tokenizer.sequence_to_ids(text)  
        input_ids, input_mask, segment_ids = input_ids

        features = {
                'input_ids': input_ids, 
                'attention_mask': input_mask, 
                'token_type_ids': segment_ids
            }
        return features

    def _convert_to_vanilla_ids(
        self, 
        text
    ):
        tokens = vanilla_tokenizer.tokenize(text)
        length = len(tokens)
        input_ids = vanilla_tokenizer.sequence_to_ids(tokens)   

        features = {
                'input_ids': input_ids,
                'length': length if length < vanilla_tokenizer.max_seq_len else vanilla_tokenizer.max_seq_len,
            }
        return features

    def _get_input_ids(
        self, 
        text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            features = self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist") 

    def _get_module_one_sample_inputs(
        self, 
        features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}
    
    def predict_one_sample(
        self, 
        text='', 
        return_label_name=True,
        return_proba=False
    ):

        features = self._get_input_ids(text)
        self.module.eval()
        
        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            logit = self.module(**inputs)
                            
        preds = logit.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        preds = preds[0][1:]
        preds = preds[:len(text)]
                
        tags = [self.id2cat[x] for x in preds]
        label_entities = get_entities(preds, self.id2cat, self.markup)
        
        entities = set()
        for entity_ in label_entities:
            entities.add(text[entity_[1]: entity_[2]+1] + '-' + entity_[0])
            
        entities = []
        for entity_ in label_entities:
            entities.append({
                "start_idx":entity_[1],
                "end_idx":entity_[2],
                "entity":text[entity_[1]: entity_[2]+1],
                "type":entity_[0]
            })
        
        return entities