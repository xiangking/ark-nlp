import time
import torch
import math
import torch.nn.functional as F
from torch import nn
from transformers import BertModel
from transformers import BertPreTrainedModel
from ark_nlp.nn.base.bert import BertForTokenClassification
from ark_nlp.nn.layer.crf_block import CRF


class CRFBert(BertForTokenClassification):
    """
    基于BERT + CRF 的命名实体模型

    :param config: (obejct) 模型的配置对象
    :param bert_trained: (bool) bert参数是否可训练，默认可训练

    :returns: 
    """ 

    def __init__(
        self, 
        config, 
        encoder_trained=True
    ):
        super(CRFBert, self).__init__(config, encoder_trained)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)