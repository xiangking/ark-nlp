import torch

from torch import nn
from transformers import BertModel
from transformers import BertPreTrainedModel


class CasRelBert(BertPreTrainedModel):
    """
    基于BERT的Casrel模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预设的文本最大长度

    Reference:
        [1] A Novel Cascade Binary Tagging Framework for Relational Triple Extraction
        [2] https://github.com/longlongman/CasRel-pytorch-reimplement
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(CasRelBert, self).__init__(config)

        self.bert = BertModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder_dim = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.sub_heads_linear = nn.Linear(self.encoder_dim, 1)
        self.sub_tails_linear = nn.Linear(self.encoder_dim, 1)
        self.obj_heads_linear = nn.Linear(self.encoder_dim, self.num_labels)
        self.obj_tails_linear = nn.Linear(self.encoder_dim, self.num_labels)

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        sub = (sub_head + sub_tail) / 2

        encoded_text = encoded_text + sub

        pred_obj_heads = self.obj_heads_linear(encoded_text)
        pred_obj_heads = torch.sigmoid(pred_obj_heads)

        pred_obj_tails = self.obj_tails_linear(encoded_text)
        pred_obj_tails = torch.sigmoid(pred_obj_tails)

        return pred_obj_heads, pred_obj_tails

    def get_subs(self, encoded_text):
        pred_sub_heads = self.sub_heads_linear(encoded_text)
        pred_sub_heads = torch.sigmoid(pred_sub_heads)

        pred_sub_tails = self.sub_tails_linear(encoded_text)
        pred_sub_tails = torch.sigmoid(pred_sub_tails)

        return pred_sub_heads, pred_sub_tails

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        sub_head=None,
        sub_tail=None,
        **kwargs
    ):

        encoded_text = self.bert(input_ids, attention_mask=attention_mask)[0]

        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        sub_head_mapping = sub_head.unsqueeze(1)
        sub_tail_mapping = sub_tail.unsqueeze(1)

        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(
            sub_head_mapping,
            sub_tail_mapping,
            encoded_text
        )

        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails
