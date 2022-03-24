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
# From: https://github.com/huggingface/transformers


import math
import torch

from typing import Tuple
from typing import Optional
from torch import nn
from torch import Tensor
from torch import device
from dataclasses import dataclass
from transformers import BertConfig
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput


@dataclass
class BaseBertModelOutput(ModelOutput):
    """
    基础Bert模型输出类，继承自`transformers.file_utils.ModelOutput`，方便兼容索引取值和属性取值
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


class BertEmbeddings(nn.Module):
    """
    bert的嵌入层，包含词嵌入、位置嵌入和token类型嵌入
    """  # noqa: ignore flake8"

    def __init__(self, config):
        super().__init__()
        
        # bert的输入分为三部分：词嵌入、位置嵌入和token类型嵌入
        #（token类型嵌入用于区分词是属于哪个句子，主要用于N个句子拼接输入的情况）
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # bert的位置嵌入使用的是绝对位置，即从句首开始按自然数进行编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 初始化时position_ids按设置中的max_position_embeddings生成，在forward会根据input_ids输入长度进行截断
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1))
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


class BertSelfAttention(nn.Module):
    """
    Bert自注意力层
    """  # noqa: ignore flake8"
    
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        # 虽然在解释多头机制的时候都是以向量直接分别经过几个NN生成
        # 但实际的实现思路类似于对向量按头的个数在维度上进行分割，然后分别经过NN后进行拼接
        # 实际代码不会显式地进行分割，但思路上是一致的
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # new_x_shape就是（批大小，序列长度, 注意力头数，每个注意力头的维度）
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        # "query"、"key"和"value"分别编码
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # "query"、"key"和"value"维度和位置调整
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # "query"和"key"的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 论文里提到的除以维度
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # 由于后续需要通过softmax，所以这里的mask并不是通过使attention_scores
            # 对应位置变为0实现的，而是将原先的mask中的0转化成-1e4，1转化成0，这样
            # 相加之后attention_scores中应该被mask的位置就变成了一个大负数
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 注意力dropout,transformers库中提到这是原始代码的实现
        attention_probs = self.dropout(attention_probs)
        
        # "attention"和"value"的点积
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 维度恢复
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class BertSelfOutput(nn.Module):
    """
    Bert自注意力输出层，本质上就是使用了self-attention之后需要经过的残差和LayerNorm层
    """  # noqa: ignore flake8"
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """
    Bert注意力层，本质上就是对自注意力层+残差+LayerNorm层的一个封装，方便调用
    """  # noqa: ignore flake8"
    
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    """
    Bert中间层，单层全连接层（升维，原始bert：768->3072）+激活函数（原始bert实现使用gelu函数）
    """  # noqa: ignore flake8"
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    Bert单层编码输出层，单层全连接层（降维，原始bert：3072->768）+dropout+残差+LayerNorm层
    """  # noqa: ignore flake8"
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    """
    Bert编码层（单层），即对前面所有层的封装，可分为三部分：
        1. bert注意力层：自注意力层+残差+LayerNorm层
        2. Bert中间层：单层全连接层+激活函数
        3. Bert单层编码输出层：单层全连接层+dropout+残差+LayerNorm层
    """  # noqa: ignore flake8"
    
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        # 注意力层
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        
        # 中间层
        intermediate_output = self.intermediate(attention_output)
        
        # 输出层
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs

        return outputs


class BertEncoder(nn.Module):
    """Bert编码器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        # N层编码过程，每层的结构都是一致的
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 该段代码是核心代码，即单层bert编码
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        # 在输出将bert最后一层的输出单独作为提取出来作为一项
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 下面都是关于返回形式的处理
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        
        return BaseBertModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BertPooler(nn.Module):
    """用于cls pooling方式的Pooler"""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModel(PreTrainedModel):
    """
    用于初始化权重和加载预训练语言模型权重的抽象类
    """  # noqa: ignore flake8"

    config_class = BertConfig
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 截断正态分布
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 截断正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class BertModel(BertPreTrainedModel):
    """
    基础的Bert模型，仅对encoder功能进行实现，并不兼容decoder功能，原始的transformers（v4.0.0）实现是兼容decoder
    """  # noqa: ignore flake8"

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        
    def get_extended_attention_mask(
        self,
        attention_mask: Tensor
    ) -> Tensor:
        """
        为了实现self-attention中的mask操作，需要将原先的attention_mask矩阵的值域从0和1变为-1e4和0
        需要注意的是：由于BertPreTrainedModel继承自PreTrainedModel，所以是带有该方法的，但为了保证
        用户可以在一个文件中就了解模型全貌，ark-nlp会将其他文件中的方法尽量copy到一个文件中，并去掉
        与该模型无关的兼容性代码
        Args:
            attention_mask (`torch.Tensor`):
                Mask矩阵，1表示可使用，0表示被遮蔽

        Returns:
            `torch.Tensor` 调整值域后的attention_mask
        """  # noqa: ignore flake8"
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
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
        
        # bert的输入分为三部分：词嵌入、位置嵌入和token类型嵌入
        # 三种嵌入向量使用`+`的方式融合到一起
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids
        )
        
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
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseBertModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )