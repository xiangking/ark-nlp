import torch.nn as nn


class Seq2SeqCrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, **kwargs):
        super().__init__(ignore_index=0, **kwargs)

    def forward(self, outputs, target):
        '''
        y_pred: [btz, seq_len, hdsz]
        targets: y_true, y_segment
        '''
        y_pred = outputs
        y_true, y_mask = target['input_ids'], target['token_type_ids']
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1, :]  # 预测序列，错开一位

        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = (y_true * y_mask).flatten()
        return super().forward(y_pred, y_true)