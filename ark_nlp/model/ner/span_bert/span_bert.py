import torch
import torch.nn.functional as F

from torch import nn
from transformers import BertModel
from ark_nlp.nn.base.bert import BertForTokenClassification
from ark_nlp.nn.layer.pooler_block import PoolerStartLogits
from ark_nlp.nn.layer.pooler_block import PoolerEndLogits


class SpanDependenceBert(BertForTokenClassification):
    """
    基于BERT指针的命名实体模型(end指针依赖start指针的结果)

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(SpanDependenceBert, self).__init__(config)

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


class SpanIndependenceBert(BertForTokenClassification):
    """
    基于BERT指针的命名实体模型(start和end指针互相不影响)

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        mid_hidden_size=128,
        mid_dropout_rate=0.1,
        **kwargs
    ):
        super(SpanIndependenceBert, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.mid_linear = nn.Sequential(
            nn.Linear(config.hidden_size, mid_hidden_size),
            nn.ReLU(),
            nn.Dropout(mid_dropout_rate)
        )

        self.start_fc = nn.Linear(mid_hidden_size, self.num_labels)
        self.end_fc = nn.Linear(mid_hidden_size, self.num_labels)

        init_blocks = [self.mid_linear, self.start_fc, self.end_fc]

        for block in init_blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

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

        sequence_output = self.mid_linear(sequence_output)

        start_logits = self.start_fc(sequence_output)

        end_logits = self.end_fc(sequence_output)

        return (start_logits, end_logits)
