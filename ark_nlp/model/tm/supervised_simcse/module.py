# Copyright (c) 2022 DataArk Authors. All Rights Reserved.
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
from ark_nlp.nn import BertMixin
from transformers import BertPreTrainedModel


class SimCSE(BertMixin, BertPreTrainedModel):
    """
    有监督的SimCSE模型
    Args:
        config:
            模型的配置对象
        encoder_trained (bool, optional):
            bert参数是否可训练, 默认值为True
        pooling (string, optional):
            bert输出的池化方式, 可选有["cls", "cls_with_pooler", "first_last_avg", "last_avg", "last_2_avg"]
            默认为"cls_with_pooler"
        dropout (float or None, optional):
            dropout比例
            默认为None, 实际运行时, 代码会将None情况下的dropout rate设置成0.1
        margin (float, optional):
            相似矩阵对角线值, 默认值为0.0
        scale (float, optional):
            相似矩阵放大倍数, 默认值为20
        output_emb_size (int, optional):
            输出的矩阵的维度
            默认值为0, 即不进行矩阵维度变换
    Reference:
        [1] SimCSE: Simple Contrastive Learning of Sentence Embeddings
        [2] https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_matching/simcse 
    """  # noqa: ignore flake8"

    def __init__(self,
                 config,
                 encoder_trained=True,
                 pooling='cls_with_pooler',
                 scale=20,
                 output_emb_size=0):

        super(SimCSE, self).__init__(config)

        self.bert = BertModel(config)
        self.pooling = pooling

        self.output_emb_size = output_emb_size
        if self.output_emb_size > 0:
            self.emb_reduce_linear = nn.Linear(config.hidden_size, self.output_emb_size)
            torch.nn.init.trunc_normal_(self.emb_reduce_linear.weight, std=0.02)

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

    def cosine_sim(self,
                   input_ids,
                   contrastive_input_ids,
                   token_type_ids=None,
                   position_ids_ids=None,
                   attention_mask=None,
                   contrastive_token_type_ids=None,
                   contrastive_position_ids=None,
                   contrastive_attention_mask=None,
                   pooling='cls_with_pooler',
                   **kwargs):

        cls_embedding = self.get_pooled_embedding(input_ids,
                                                  token_type_ids,
                                                  position_ids_ids,
                                                  attention_mask,
                                                  pooling=pooling)

        contrastive_cls_embedding = self.get_pooled_embedding(contrastive_input_ids,
                                                              contrastive_token_type_ids,
                                                              contrastive_position_ids,
                                                              contrastive_attention_mask,
                                                              pooling=pooling)

        cosine_sim = torch.sum(cls_embedding * contrastive_cls_embedding, axis=-1)

        return cosine_sim

    def forward(self,
                input_ids,
                contrastive_input_ids,
                negative_input_ids=None,
                attention_mask=None,
                contrastive_attention_mask=None,
                negative_attention_mask=None,
                token_type_ids=None,
                contrastive_token_type_ids=None,
                negative_token_type_ids=None,
                position_ids=None,
                contrastive_position_ids=None,
                negative_position_ids=None,
                **kwargs):

        if negative_input_ids is not None:
            contrastive_input_ids = torch.concat(
                [contrastive_input_ids, negative_input_ids])

        if negative_attention_mask is not None:
            contrastive_attention_mask = torch.concat(
                [contrastive_attention_mask, negative_attention_mask])

        if negative_token_type_ids is not None:
            contrastive_token_type_ids = torch.concat(
                [contrastive_token_type_ids, negative_token_type_ids])

        if negative_position_ids is not None:
            contrastive_position_ids = torch.concat(
                [contrastive_position_ids, negative_position_ids])

        cls_embedding = self.get_pooled_embedding(input_ids, token_type_ids, position_ids,
                                                  attention_mask)

        contrastive_cls_embedding = self.get_pooled_embedding(contrastive_input_ids,
                                                              contrastive_token_type_ids,
                                                              contrastive_position_ids,
                                                              contrastive_attention_mask)

        cosine_sim = torch.matmul(cls_embedding, contrastive_cls_embedding.T)

        cosine_sim *= self.sacle

        labels = torch.arange(0, cls_embedding.shape[0])
        labels = torch.reshape(labels, shape=[-1, 1]).squeeze(1)

        loss = F.cross_entropy(cosine_sim, labels.to(cls_embedding.device))

        return cosine_sim, loss
