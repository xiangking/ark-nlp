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

from torch import nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from ark_nlp.nn.base.bert import Bert


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.num_labels))

        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertForPromptMaskedLM(Bert):
    """
    针对prompt的基于BERT的mlm任务

    :param config: (obejct) 模型的配置对象
    :param bert_trained: (bool) bert参数是否可训练, 默认可训练

    :returns:

    Reference:
        [1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(BertForPromptMaskedLM, self).__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.classifier = BertLMPredictionHead(config)

        self.init_weights()

    @staticmethod
    def _batch_gather(data: torch.Tensor, index: torch.Tensor):
        """
        实现类似tf.batch_gather的效果

        Args:
            data (torch.Tensor): (bs, max_seq_len, hidden)的张量
            index (torch.Tensor): (bs, n)需要gather的位置

        Returns:
            torch.Tensor: (bs, n, hidden)的张量
        """
        # (bs, n, hidden)
        index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)

        return torch.gather(data, 1, index)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_position=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        sequence_output = BertForPromptMaskedLM._batch_gather(sequence_output, mask_position)

        batch_size, _, hidden_size = sequence_output.shape

        sequence_output = sequence_output.reshape(-1, hidden_size)

        out = self.classifier(sequence_output)

        return out
