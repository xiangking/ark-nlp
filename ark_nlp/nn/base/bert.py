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

import time
import torch
import math
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from ark_nlp.nn import BasicModule
from transformers import BertModel
from transformers import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from ark_nlp.nn.layer.crf_block import CRF


class VanillaBert(BertPreTrainedModel):
    """
    原始的BERT模型

    :param config: (obejct) 模型的配置对象
    :param bert_trained: (bool) bert参数是否可训练，默认可训练

    :returns: 

    Reference:
        [1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
    """ 
    def __init__(
        self, 
        config, 
        encoder_trained=True,
        pooling='cls'
    ):
        super(VanillaBert, self).__init__(config)
        self.bert = BertModel(config)
        self.pooling = pooling
        
        for param in self.bert.parameters():
            param.requires_grad = encoder_trained 
            
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        self.init_weights()
        
    def mask_pooling(self, x: Tensor, attention_mask=None):
        if attention_mask is None:
            return torch.mean(x, dim=1)
        return torch.sum(x * attention_mask.unsqueeze(2), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)

    def sequence_pooling(self, sequence_feature, attention_mask):
        if self.pooling == 'first_last_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[1]
        elif self.pooling == 'last_avg':
            sequence_feature = sequence_feature[-1]
        elif self.pooling == 'last_2_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[-2]
        elif self.pooling == 'cls':
            return sequence_feature[-1][:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(self.pooling))

        return self.mask_pooling(sequence_feature, attention_mask)

    def get_encoder_feature(self, encoder_output, attention_mask):
        if self.task == 'SequenceLevel':
            return self.sequence_pooling(encoder_output, attention_mask)
        elif self.task == 'TokenLevel':
            return encoder_output[-1]
        else:
            return encoder_output[-1][:, 0, :]

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True, 
                            output_hidden_states=True
                           ).hidden_states

        encoder_feature = self.get_encoder_feature(outputs, attention_mask)

        encoder_feature = self.dropout(encoder_feature)
        out = self.classifier(encoder_feature)

        return out


class Bert(BertPreTrainedModel):
    """
    原始的BERT模型

    :param config: (obejct) 模型的配置对象
    :param bert_trained: (bool) bert参数是否可训练，默认可训练

    :returns: 

    Reference:
        [1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
    """ 
    def __init__(
        self, 
        config, 
        encoder_trained=True,
        pooling='cls'
    ):
        super(Bert, self).__init__(config)
        
        self.bert = BertModel(config)
        self.pooling = pooling
        
        for param in self.bert.parameters():
            param.requires_grad = encoder_trained 
            
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        self.init_weights()
        
    def mask_pooling(self, x: Tensor, attention_mask=None):
        if attention_mask is None:
            return torch.mean(x, dim=1)
        return torch.sum(x * attention_mask.unsqueeze(2), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)

    def sequence_pooling(self, sequence_feature, attention_mask):
        if self.pooling == 'first_last_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[1]
        elif self.pooling == 'last_avg':
            sequence_feature = sequence_feature[-1]
        elif self.pooling == 'last_2_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[-2]
        elif self.pooling == 'cls':
            return sequence_feature[-1][:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(self.pooling))

        return self.mask_pooling(sequence_feature, attention_mask)

    def get_encoder_feature(self, encoder_output, attention_mask):
        if self.task == 'SequenceLevel':
            return self.sequence_pooling(encoder_output, attention_mask)
        elif self.task == 'TokenLevel':
            return encoder_output[-1]
        else:
            return encoder_output[-1][:, 0, :]

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True, 
                            output_hidden_states=True
                           ).hidden_states

        encoder_feature = self.get_encoder_feature(outputs, attention_mask)

        encoder_feature = self.dropout(encoder_feature)
        out = self.classifier(encoder_feature)

        return out


class BertForSequenceClassification(BertPreTrainedModel):
    """
    基于BERT的文本分类模型

    :param config: (obejct) 模型的配置对象
    :param bert_trained: (bool) bert参数是否可训练，默认可训练

    :returns: 

    Reference:
        [1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  

    """ 
    def __init__(
        self, 
        config, 
        encoder_trained=True
    ):
        super(BertForSequenceClassification, self).__init__(config)
        
        self.bert = BertModel(config)
        
        for param in self.bert.parameters():
            param.requires_grad = encoder_trained 
            
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        self.init_weights()

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids) 

        sequence_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        out = self.classifier(sequence_output)

        return out


class BertForTokenClassification(BertPreTrainedModel):
    """
    基于BERT的命名实体模型

    :param config: (obejct) 模型的配置对象
    :param bert_trained: (bool) bert参数是否可训练，默认可训练

    :returns: 

    Reference:
        [1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  

    """ 
    def __init__(
        self, 
        config, 
        encoder_trained=True
    ):
        super(BertForTokenClassification, self).__init__(config)
        
        self.bert = BertModel(config)
        
        for param in self.bert.parameters():
            param.requires_grad = encoder_trained 
            
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        self.init_weights()

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids) 

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        out = self.classifier(sequence_output)

        return out