import time
import torch
import math
import torch.nn.functional as F
from torch import nn
from ark_nlp.nn.base.bert import BertForTokenClassification
from ark_nlp.nn.layer.crf_block import CRF


class CRFErnie(BertForTokenClassification):
    """
    基于Ernie + CRF 的命名实体模型

    :param config: (obejct) 模型的配置对象
    :param encoder_trained: (bool) 预训练模型的参数是否可训练，默认可训练

    :returns: 
    """ 

    def __init__(
        self, 
        config, 
        encoder_trained=True
    ):
        super(CRFErnie, self).__init__(config, encoder_trained)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)