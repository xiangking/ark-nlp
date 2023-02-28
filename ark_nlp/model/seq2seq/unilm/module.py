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
from torch import Tensor
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertPreTrainedModel

from ark_nlp.nn.base.bert import BertMixin
from ark_nlp.nn.base.roformer import RoFormerPreTrainedModel, RoFormerModel


class BertGenerationOnlyLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        logits = self.decoder(hidden_states)
        return logits


class UniLMBert(BertMixin, BertPreTrainedModel):
    """
    原始的RoFormer模型

    Args:
        config:
            模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练
        pooling (:obj:`str`, optional, defaults to "cls_with_pooler"):
            bert输出的池化方式，默认为"cls_with_pooler"，
            可选有["cls", "cls_with_pooler", "first_last_avg", "last_avg", "last_2_avg"]

    Reference:
        [1] https://github.com/ZhuiyiTechnology/roformer
        [2] https://github.com/JunnYu/RoFormer_pytorch
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        pooling='cls_with_pooler',
    ):
        super(UniLMBert, self).__init__(config)

        self.bert = BertModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.classifier = BertGenerationOnlyLMHead(config)

        self.init_weights()

    def compute_attention_bias(self, segment_ids):
        idxs = torch.cumsum(segment_ids, dim=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        mask = mask.to(torch.float)
        mask = -(1.0 - mask) * 1e12
        return mask

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        **kwargs
    ):
        extended_attention_mask = self.compute_attention_bias(token_type_ids)
        outputs = self.bert(
            input_ids,
            attention_mask=extended_attention_mask,
        )

        encoder_feature = outputs[0]

        out = self.classifier(encoder_feature)

        return out