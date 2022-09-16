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
# Author: Chenjie Shen, jimme.shen123@gmail.com
# Status: Active


import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertPreTrainedModel

from ark_nlp.nn.base.bert import BertMixin


class SBert(BertMixin, BertPreTrainedModel):
    """
    SBert模型

    Args:
        config:
            模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练
        pooling (:obj:`str`, optional, defaults to "cls"):
            bert输出的池化方式，默认为"cls_with_pooler"，
            可选有["cls", "cls_with_pooler", "first_last_avg", "last_avg", "last_2_avg"]
        output_emb_size (:obj:`int`, optional, defaults to 0):
            输出的矩阵的维度，默认为0，即不进行矩阵维度变换

    Reference:
        [1] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
        [2] https://github.com/abdouaziz/SBert
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        pooling='cls',
        dropout=None,
        output_emb_size=0
    ):

        super(SBert, self).__init__(config)

        self.bert = BertModel(config)
        self.pooling = pooling

        self.out = nn.Linear(config.hidden_size * 3, config.num_labels)

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
        do_normalize=True,
    ):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            return_dict=True,
                            output_hidden_states=True)

        encoder_feature = self.get_encoder_feature(outputs, attention_mask, pooling)

        if self.output_emb_size > 0:
            encoder_feature = self.emb_reduce_linear(encoder_feature)

        if do_normalize is True:
            out = F.normalize(encoder_feature, p=2, dim=-1)

        return out

    def cosine_sim(
        self,
        input_ids_a,
        input_ids_b,
        token_type_ids_a=None,
        position_ids_a=None,
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
            position_ids_a,
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

        cosine_sim = torch.sum(query_cls_embedding * title_cls_embedding, axis=-1)

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

        u_v = torch.abs(cls_embedding_a - cls_embedding_b)

        concat = torch.cat((cls_embedding_a, cls_embedding_b, u_v), dim=-1)

        out = self.out(concat)

        return out
