# Copyright (c) 2021 DataArk Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xiang Wang, xiangking1995@163.com
# Status: Active


import torch
import torch.nn.functional as F

from torch import nn
from transformers import BertModel
from ark_nlp.nn import Bert


class SimCSE(Bert):
    """
    无监督的SimCSE模型

    Args:
        config:
            模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练
        pooling (:obj:`str`, optional, defaults to "cls"):
            bert输出的池化方式，默认为"cls_with_pooler"，
            可选有["cls", "cls_with_pooler", "first_last_avg", "last_avg", "last_2_avg"]
        dropout (:obj:`float` or :obj:`None`, optional, defaults to None):
            dropout比例，默认为None，实际设置时会设置成0.1
        margin (:obj:`float`, optional, defaults to 0.0):
            相似矩阵对角线值
        scale (:obj:`float`, optional, defaults to 20):
            相似矩阵放大倍数
        output_emb_size (:obj:`int`, optional, defaults to 0):
            输出的矩阵的维度，默认为0，即不进行矩阵维度变换

    Reference:
        [1] SimCSE: Simple Contrastive Learning of Sentence Embeddings
        [2] https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_matching/simcse 
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        pooling='cls_with_pooler',
        dropout=None,
        margin=0.0,
        scale=20,
        output_emb_size=0
    ):

        super(SimCSE, self).__init__(config)

        self.bert = BertModel(config)
        self.pooling = pooling

        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # if output_emb_size is greater than 0, then add Linear layer to reduce embedding_size,
        # we recommend set output_emb_size = 256 considering the trade-off beteween
        # recall performance and efficiency
        self.output_emb_size = output_emb_size
        if self.output_emb_size > 0:
            self.emb_reduce_linear = nn.Linear(
                config.hidden_size,
                self.output_emb_size
            )
            torch.nn.init.trunc_normal_(
                self.emb_reduce_linear.weight,
                std=0.02
            )

        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.sacle = scale

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.init_weights()

    def get_pooled_embedding(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        pooling='cls_with_pooler',
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True
        )

        encoder_feature = self.get_encoder_feature(
            outputs,
            attention_mask,
            pooling
        )

        if self.output_emb_size > 0:
            encoder_feature = self.emb_reduce_linear(encoder_feature)

        encoder_feature = self.dropout(encoder_feature)
        out = F.normalize(encoder_feature, p=2, dim=-1)

        return out

    def cosine_sim(
        self,
        input_ids_a,
        input_ids_b,
        token_type_ids_a=None,
        position_ids_ids_a=None,
        attention_mask_a=None,
        token_type_ids_b=None,
        position_ids_b=None,
        attention_mask_b=None,
        pooling='cls',
        **kwargs
    ):

        query_cls_embedding = self.get_pooled_embedding(
            input_ids_a,
            token_type_ids_a,
            position_ids_ids_a,
            attention_mask_a,
            pooling=pooling
        )

        title_cls_embedding = self.get_pooled_embedding(
            input_ids_b,
            token_type_ids_b,
            position_ids_b,
            attention_mask_b,
            pooling=pooling
        )

        cosine_sim = torch.sum(
            query_cls_embedding * title_cls_embedding,
            axis=-1
        )

        return cosine_sim

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        token_type_ids_a=None,
        position_ids_ids_a=None,
        attention_mask_a=None,
        token_type_ids_b=None,
        position_ids_b=None,
        attention_mask_b=None,
        **kwargs
    ):

        cls_embedding_a = self.get_pooled_embedding(
            input_ids_a,
            token_type_ids_a,
            position_ids_ids_a,
            attention_mask_a
        )

        cls_embedding_b = self.get_pooled_embedding(
            input_ids_b,
            token_type_ids_b,
            position_ids_b,
            attention_mask_b
        )

        cosine_sim = torch.matmul(cls_embedding_a, cls_embedding_b.T)

        cosine_sim = cosine_sim - torch.eye(cls_embedding_a.shape[0], device=cls_embedding_a.device) * self.margin

        # 论文中使用的是除以0.05，这边只是把它改成乘以扩大倍数，更好理解
        cosine_sim *= self.sacle

        labels = torch.arange(0, cls_embedding_a.shape[0])
        labels = torch.reshape(labels, shape=[-1, 1]).squeeze(1)

        loss = F.cross_entropy(cosine_sim, labels.to(input_ids_a.device))

        return cosine_sim, loss
