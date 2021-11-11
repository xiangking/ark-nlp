from ark_nlp.nn.base.bert import BertForTokenClassification
from ark_nlp.nn.layer.crf_block import CRF


class CrfBert(BertForTokenClassification):
    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        """
        初始化基于BERT + CRF 的命名实体模型

        Args:
            config: 模型的配置对象
            bert_trained (:obj:`bool`, optional): 预设的文本最大长度
        """  # noqa: ignore flake8"

        super(CrfBert, self).__init__(config, encoder_trained)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
