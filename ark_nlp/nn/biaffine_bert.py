import torch

from torch import nn
from transformers import BertModel
from ark_nlp.nn.base.bert import BertForTokenClassification
from ark_nlp.nn.layer.biaffine_block import Biaffine


class BiaffineBert(BertForTokenClassification):
    """
    Biaffine的命名实体识别模型

    Args:
        config: 模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练
        biaffine_size (:obj:`int`, optional, defaults to 128): 
            biaffine输入的embedding size
        lstm_dropout (:obj:`float`, optional, defaults to 0.4): 
            lstm的dropout rate
        select_bert_layer (:obj:`int`, optional, defaults to -1): 
            获取哪一层的bert embedding

    Reference:
        [1] Named Entity Recognition as Dependency Parsing
        [2] https://github.com/suolyer/PyTorch_BERT_Biaffine_NER
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        biaffine_size=128,
        lstm_dropout=0.4,
        select_bert_layer=-1
    ):
        super(BiaffineBert, self).__init__(config)

        self.num_labels = config.num_labels
        self.select_bert_layer = select_bert_layer

        self.bert = BertModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.lstm = torch.nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )

        self.start_encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2*config.hidden_size,
                out_features=biaffine_size),
            torch.nn.ReLU()
        )

        self.end_encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2*config.hidden_size,
                out_features=biaffine_size),
            torch.nn.ReLU()
        )

        self.biaffne = Biaffine(biaffine_size, self.num_labels)

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.start_encoder[0].weight)
        nn.init.xavier_uniform_(self.end_encoder[0].weight)

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
        )

        sequence_output = outputs.hidden_states[self.select_bert_layer]

        # lstm编码
        sequence_output, _ = self.lstm(sequence_output)

        start_logits = self.start_encoder(sequence_output)
        end_logits = self.end_encoder(sequence_output)

        span_logits = self.biaffne(start_logits, end_logits)
        span_logits = span_logits.contiguous()

        return span_logits
