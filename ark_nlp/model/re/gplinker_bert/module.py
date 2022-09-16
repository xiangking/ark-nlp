from transformers import BertModel
from transformers import BertPreTrainedModel
from ark_nlp.nn.layer.global_pointer_block import GlobalPointer
from ark_nlp.nn.layer.global_pointer_block import EfficientGlobalPointer


class GPLinkerBert(BertPreTrainedModel):
    """
    GPLinker + Bert 的联合抽取模型

    Args:
        config: 模型的配置对象
        bert_trained (bool, optional): 预训练模型的参数是否可训练
        entity_type_num (int, optional): 实体类型, 默认值为2, 即头实体和尾实体
        relation_type_num (int, optional): 关系类型, 一般是使用头实体类型+关系类型+尾实体类型, 默认值为2
        head_size (int, optional): GlobalPointer head的数目, 默认值为64

    Reference:
        [1] https://www.kexue.fm/archives/8888
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        entity_type_num=2,
        relation_type_num=2,
        head_size=64, 
    ):
        super(GPLinkerBert, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.entity_global_pointer = GlobalPointer(
            entity_type_num,
            head_size,
            config.hidden_size
        )
        
        self.head_global_pointer = GlobalPointer(
            relation_type_num,
            head_size,
            config.hidden_size,
            RoPE=False,
            tril_mask=False
        )
        
        self.tail_global_pointer = GlobalPointer(
            relation_type_num,
            head_size,
            config.hidden_size,
            RoPE=False,
            tril_mask=False
        )

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

        entity_logits = self.entity_global_pointer(sequence_output, mask=attention_mask)
        head_logits = self.head_global_pointer(sequence_output, mask=attention_mask)
        tail_logits = self.tail_global_pointer(sequence_output, mask=attention_mask)

        return entity_logits, head_logits, tail_logits