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

import torch

from torch import nn
from torch import Tensor
from transformers import BertModel
from transformers import BertPreTrainedModel


class Bert(BertPreTrainedModel):
    """
    原始的BERT模型

    Args:
        config:
            模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练
        pooling (:obj:`str`, optional, defaults to "cls"):
            bert输出的池化方式，默认为"cls_with_pooler"，
            可选有["cls", "cls_with_pooler", "first_last_avg", "last_avg", "last_2_avg"]

    Reference:
        [1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        pooling='cls_with_pooler'
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

    def sequence_pooling(self, sequence_feature, attention_mask, pooling):

        if pooling is None:
            pooling = self.pooling

        if pooling == 'cls_with_pooler':
            return sequence_feature.pooler_output
        sequence_feature = sequence_feature.hidden_states
        if pooling == 'first_last_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[1]
        elif pooling == 'last_avg':
            sequence_feature = sequence_feature[-1]
        elif pooling == 'last_2_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[-2]
        elif pooling == 'cls':
            return sequence_feature[-1][:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(pooling))

        return self.mask_pooling(sequence_feature, attention_mask)

    def get_encoder_feature(
        self,
        encoder_output,
        attention_mask,
        pooling=None
    ):
        if self.task == 'SequenceLevel':
            return self.sequence_pooling(
                encoder_output,
                attention_mask,
                pooling
            )
        elif self.task == 'TokenLevel':
            return encoder_output[-1]
        else:
            return encoder_output[-1][:, 0, :]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True
        )

        encoder_feature = self.get_encoder_feature(outputs, attention_mask)

        encoder_feature = self.dropout(encoder_feature)
        out = self.classifier(encoder_feature)

        return out


class BertForSequenceClassification(BertPreTrainedModel):
    """
    基于BERT的文本分类模型

    Args:
        config:
            模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练

    Reference:
        [1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
    """  # noqa: ignore flake8"

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
        position_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        sequence_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        out = self.classifier(sequence_output)

        return out


class BertForTokenClassification(BertPreTrainedModel):
    """
    基于BERT的命名实体模型

    Args:
        config:
            模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练

    Reference:
        [1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
    """  # noqa: ignore flake8"

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
        position_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        out = self.classifier(sequence_output)

        return out
