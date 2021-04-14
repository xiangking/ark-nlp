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
from .basemodel import BasicModule
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss


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
        bert_trained=True
    ):
        super(VanillaBert, self).__init__(config)
        
        self.bert = BertModel(config)
        
        for param in self.bert.parameters():
            param.requires_grad = bert_trained 
            
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        self.init_weights()

    def get_encoder_feature(self, encoder_output):
        if self.task == 'SequenceClassification':
            return encoder_output[1]
        elif self.task == 'TokenClassification':
            return encoder_output[0]
        else:
            return encoder_output[1]

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

        encoder_feature = self.get_encoder_feature(outputs)

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
        bert_trained=True
    ):
        super(BertForTextClassification, self).__init__(config)
        
        self.bert = BertModel(config)
        
        for param in self.bert.parameters():
            param.requires_grad = bert_trained 
            
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
        bert_trained=True
    ):
        super(BertForTextClassification, self).__init__(config)
        
        self.bert = BertModel(config)
        
        for param in self.bert.parameters():
            param.requires_grad = bert_trained 
            
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
