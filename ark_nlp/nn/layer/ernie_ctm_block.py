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

from dataclasses import dataclass
from typing import Tuple
from typing import Optional
from torch import nn
from torch import Tensor
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertPooler
from transformers.models.bert.modeling_bert import BertEncoder


@dataclass
class BaseBertModelOutput(ModelOutput):
    """
    基础Bert模型输出类, 继承自`transformers.file_utils.ModelOutput`, 方便兼容索引取值和属性取值
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            bert编码器最后一层的输出向量
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            CLS向量经过BertPooler之后的输出
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            bert模型每层的输出向量
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            所有attention softmax后的注意力值
    """  # noqa: ignore flake8"

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ErnieCtmEmbeddings(nn.Module):
    """
    ErnieCtm的嵌入层, 包含词嵌入、位置嵌入和token类型嵌入,
    与原版相比, 修改了position_ids的默认设置, 增加了根据cls类特殊字符的数量设置0位置的功能
    """

    def __init__(self, config):
        super().__init__()

        self.cls_num = config.cls_num

        # bert的输入分为三部分：词嵌入、位置嵌入和token类型嵌入
        # （token类型嵌入用于区分词是属于哪个句子，主要用于N个句子拼接输入的情况）
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # bert的位置嵌入使用的是绝对位置，即从句首开始按自然数进行编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 初始化时position_ids按设置中的max_position_embeddings生成，在forward会根据input_ids输入长度进行截断
        # 由于ErnieCtm类模型会增加[CLS]的数量，所以需要将默认的位置向量的0位置扩充
        self.register_buffer(
            "position_ids",
            torch.cat([torch.zeros(size=[self.cls_num], dtype=torch.int64), torch.arange(1, config.max_position_embeddings)]).expand((1, -1))
        )
        # 初始化时token_type_ids按position_ids的size生成，在forward会根据input_ids输入长度进行截断
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None
    ):
        # transformers的库允许不输入input_ids而是输入向量
        # 在ark-nlp中不需要对输入向量进行兼容，ark-nlp倾向于用户自己去定义包含该功能的模型
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

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
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ErnieCtmModel(BertPreTrainedModel):
    """
    基础的Bert模型, 仅对encoder功能进行实现, 并不兼容decoder功能, 原始的transformers(v4.0.0)实现是兼容decoder
    """  # noqa: ignore flake8"

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = ErnieCtmEmbeddings(config)

        self.embedding_hidden_mapping_in = nn.Linear(
            self.config.embedding_size,
            self.config.hidden_size
        )

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.use_content_summary = config.use_content_summary
        self.content_summary_index = config.content_summary_index
        if self.use_content_summary is True:
            self.feature_fuse = nn.Linear(self.config.hidden_size * 2, self.config.intermediate_size)
            self.feature_output = nn.Linear(self.config.intermediate_size, self.config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor
    ):
        """
        为了实现self-attention中的mask操作, 需要将原先的attention_mask矩阵的值域从0和1变为-1e4和0
        需要注意的是: 由于BertPreTrainedModel继承自PreTrainedModel, 所以是带有该方法的, 但为了保证
        用户可以在一个文件中就了解模型全貌, ark-nlp会将其他文件中的方法尽量copy到一个文件中, 并去掉
        与该模型无关的兼容性代码
        Args:
            attention_mask (`torch.Tensor`):
                Mask矩阵, 1表示可使用, 0表示被遮蔽
        Returns:
            `torch.Tensor` 调整值域后的attention_mask
        """  # noqa: ignore flake8"
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("The input_ids is required")

        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 调整attention_mask的值域，将原先的attention_mask矩阵的值域从0和1变为-1e4和0
        # 以满足BertSelfAttention类的需求
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask)

        # bert的输入分为三部分: 词嵌入、位置嵌入和token类型嵌入
        # 三种嵌入向量使用`+`的方式融合到一起
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids
        )

        # ErnieCtm的embedding是128, 所以这里有个升级维度的变换
        embedding_output = self.embedding_hidden_mapping_in(embedding_output)

        # 使用Bert encoder进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # encoder_outputs[0]即last_hidden_state
        sequence_output = encoder_outputs[0]

        # 提供了cls pooling方式
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        content_output = (sequence_output[:, self.content_summary_index] if self.use_content_summary else None)

        if self.use_content_summary is True:

            content_output = content_output.unsqueeze(1).repeat(1, sequence_output.shape[1], 1)

            sequence_output = torch.cat(
                (sequence_output, content_output), 2)

            sequence_output = self.feature_fuse(sequence_output)

            sequence_output = self.feature_output(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseBertModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ErnieCtmNptagModel(BertPreTrainedModel):
    """ErnieCtmNptag模型, 用于LM预测的"""

    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(ErnieCtmNptagModel, self).__init__(config)

        self.bert = ErnieCtmModel(config)
        self.predictions = BertLMPredictionHead(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        **kwargs
    ):
        encoder_feature = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        encoder_feature = encoder_feature[0]
        logits = self.predictions(encoder_feature)

        return logits
