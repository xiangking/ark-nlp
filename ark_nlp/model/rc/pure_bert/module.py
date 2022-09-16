import torch

from torch import nn
from transformers import BertModel
from transformers import BertPreTrainedModel
from collections import Counter


class PUREBert(BertPreTrainedModel):
    """
    PURE Bert命名实体模型

    Args:
        config: 模型的配置对象
        use_rope (bool, optional): 是否使用相对位置编码, 默认值为: True

    Reference:
        [1] A Frustratingly Easy Approach for Joint Entity and Relation Extraction
        [2] https://github.com/yeqingzhao/relation-extraction
    """  # noqa: ignore flake8"

    def __init__(self, config, use_rope=True):
        super(PUREBert, self).__init__(config)
        self.bert = BertModel(config)

        self.linear_dim = int(config.hidden_size / config.num_attention_heads)
        self.linear = nn.Linear(config.hidden_size, self.linear_dim * 2)

        self.num_labels = config.num_labels
        self.classifier = nn.Linear(self.linear_dim * 4, self.num_labels)

        self.use_rope = use_rope

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        relations_idx,
        **kwargs
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids)

        last_hidden_state = outputs.last_hidden_state
        # outputs: (batch_size, seq_len, linear_dim*2)
        outputs = self.linear(last_hidden_state)
        # query, key: (batch_size, ent_type_size, seq_len, linear_dim)
        query, key = torch.split(outputs, self.linear_dim, dim=-1)

        if self.use_rope:
            indices = torch.arange(0, self.linear_dim // 2, dtype=torch.float)
            indices = torch.pow(10000, -2 * indices / self.linear_dim).to(self.device)
            position_embeddings = position_ids[..., None] * indices[None, None, :]
            # sin_embeddings:(batch_size,seg_len,linear_dim)
            sin_embeddings = torch.sin(position_embeddings).repeat_interleave(2, dim=-1)
            # cos_embeddings:(batch_size,seg_len,linear_dim)
            cos_embeddings = torch.cos(position_embeddings).repeat_interleave(2, dim=-1)

            qw2 = torch.stack([-query[..., 1::2], query[..., ::2]], -1)
            qw2 = qw2.reshape(query.shape)
            query = query * cos_embeddings + qw2 * sin_embeddings

            kw2 = torch.stack([-key[..., 1::2], key[..., ::2]], -1)
            kw2 = kw2.reshape(key.shape)
            key = key * cos_embeddings + kw2 * sin_embeddings

        # 取sub、obj的emb
        batch_idx, sub_start_idx_list, sub_end_idx_list, obj_start_idx_list, obj_end_idx_list = [], [], [], [], []
        for i, rel_idx in enumerate(relations_idx):
            for r_idx in rel_idx:
                batch_idx.append(i)
                sub_start_idx_list.append(r_idx[0])
                sub_end_idx_list.append(r_idx[1])
                obj_start_idx_list.append(r_idx[2])
                obj_end_idx_list.append(r_idx[3])

        sub_start_query = query[batch_idx, sub_start_idx_list, :]
        sub_end_query = query[batch_idx, sub_end_idx_list, :]
        sub_query = torch.cat([sub_start_query, sub_end_query], dim=-1)

        obj_start_key = key[batch_idx, obj_start_idx_list, :]
        obj_end_key = key[batch_idx, obj_end_idx_list, :]
        obj_key = torch.cat([obj_start_key, obj_end_key], dim=-1)

        out = self.classifier(torch.cat([sub_query, obj_key], dim=-1))

        return out
