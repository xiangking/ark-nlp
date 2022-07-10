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

from ark_nlp.nn.base.bert import Bert
from ark_nlp.nn.layer.ernie_ctm_block import ErnieCtmModel
from ark_nlp.nn.layer.ernie_ctm_block import BertLMPredictionHead


class PromptErnieCtmNptag(Bert):

    """
    基于ErnieCtmNptag的prompt模型

    Args:
        config:
            模型的配置对象
        encoder_trained (bool, optional):
            bert参数是否可训练, 默认值为可训练

    Reference:
        [1] https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_to_knowledge/nptag
    """
    def __init__(
            self,
            config,
            encoder_trained=True
    ):
        super(PromptErnieCtmNptag, self).__init__(config)

        self.bert = ErnieCtmModel(config)

        self.predictions = BertLMPredictionHead(config)

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
        index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)  # (bs, n, hidden)

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

        sequence_output = PromptErnieCtmNptag._batch_gather(sequence_output, mask_position)

        batch_size, _, hidden_size = sequence_output.shape

        sequence_output = sequence_output.reshape(-1, hidden_size)

        out = self.predictions(sequence_output)

        return out
