import torch
import torch.nn.functional as F

from torch import nn
from transformers import BertModel
from ark_nlp.nn.base.bert import BertForTokenClassification
from ark_nlp.nn.layer.pooler_block import PoolerStartLogits
from ark_nlp.nn.layer.pooler_block import PoolerEndLogits


class SpanBert(BertForTokenClassification):
    """
    基于span模式的BERT指针的命名实体模型

    Args:
        config: 
            模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(SpanBert, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True
        ).hidden_states

        sequence_output = outputs[-1]

        sequence_output = self.dropout(sequence_output)

        start_logits = self.start_fc(sequence_output)

        label_logits = F.softmax(start_logits, -1)
        label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()

        end_logits = self.end_fc(sequence_output, label_logits)

        return (start_logits, end_logits)
