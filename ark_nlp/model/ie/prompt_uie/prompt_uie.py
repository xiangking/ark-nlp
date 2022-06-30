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
import torch.nn as nn

from transformers import BertModel
from transformers import BertPreTrainedModel


class BertEmbeddings(torch.nn.Module):
    """
    bert的嵌入层，包含词嵌入、位置嵌入和token类型嵌入
    """# noqa: ignore flake8"

    def __init__(self, config):
        super().__init__()

        self.use_task_id = config.use_task_id

        # bert的输入分为三部分：词嵌入、位置嵌入和token类型嵌入
        # （token类型嵌入用于区分词是属于哪个句子，主要用于N个句子拼接输入的情况）
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        if self.use_task_id:
            self.task_type_embeddings = nn.Embedding(
                config.task_type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # bert的位置嵌入使用的是绝对位置，即从句首开始按自然数进行编码
        self.position_embedding_type = getattr(config,
                                               "position_embedding_type",
                                               "absolute")
        # 初始化时position_ids按设置中的max_position_embeddings生成，在forward会根据input_ids输入长度进行截断
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)))
        # 初始化时token_type_ids按position_ids的size生成，在forward会根据input_ids输入长度进行截断
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

        if self.use_task_id:
            self.register_buffer(
                "task_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                task_type_ids=None,
                **kwargs):
        # transformers的库允许不输入input_ids而是输入向量
        # 在ark-nlp中不需要对输入向量进行兼容，ark-nlp倾向于用户自己去定义包含该功能的模型
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape,
                                             dtype=torch.long,
                                             device=self.position_ids.device)

        if task_type_ids is None:
            if hasattr(self, "task_type_ids"):
                buffered_task_type_ids = self.task_type_ids[:, :seq_length]
                buffered_task_type_ids_expanded = buffered_task_type_ids.expand(
                    input_shape[0], seq_length)
                task_type_ids = buffered_task_type_ids_expanded
            else:
                task_type_ids = torch.zeros(input_shape,
                                            dtype=torch.long,
                                            device=self.position_ids.device)

        # 生成词嵌入向量
        input_embedings = self.word_embeddings(input_ids)
        # 生成token类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # 生成位置嵌入向量
        # 此处保留transformers里的代码形式，但该判断条件对本部分代码并无实际意义
        # 本部分的位置编码仅使用绝对编码
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)

        # 将三个向量相加
        embeddings = input_embedings + position_embeddings + token_type_embeddings

        # 生成任务嵌入向量
        if self.use_task_id:
            task_type_embeddings = self.task_type_embeddings(task_type_ids)
            embeddings = embeddings + task_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class PromptUIE(BertPreTrainedModel):
    """
    通用信息抽取 UIE(Universal Information Extraction), 基于MRC结构实现

    Args:
        config: 模型的配置对象
        encoder_trained (bool, optional): bert参数是否可训练, 默认值为True

    Reference:
        [1] https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie
    """  # noqa: ignore flake8"

    def __init__(self,
                 config,
                 encoder_trained=True):
        super(PromptUIE, self).__init__(config)
        self.bert = BertModel(config)

        self.bert.embeddings = BertEmbeddings(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.num_labels = config.num_labels

        self.start_linear = torch.nn.Linear(config.hidden_size, 1)
        self.end_linear = torch.nn.Linear(config.hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,
                input_ids,
                token_type_ids=None,
                pos_ids=None,
                attention_mask=None,
                **kwargs):
        sequence_feature = self.bert(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     return_dict=True,
                                     output_hidden_states=True).hidden_states

        sequence_feature = sequence_feature[-1]

        start_logits = self.start_linear(sequence_feature)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)

        end_logits = self.end_linear(sequence_feature)

        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)

        return start_prob, end_prob
