import torch
import numpy as np

from torch import nn


class GPLinkerCrossEntropy(nn.Module):
    """
    适配GPLinker的交叉熵损失函数（稀疏多标签交叉熵损失）

    Reference:
        [1] https://kexue.fm/archives/7359
    """  # noqa: ignore flake8"
    
    def __init__(self, ):
        super(GPLinkerCrossEntropy, self).__init__()

    @staticmethod
    def sparse_multilabel_categorical_crossentropy(y_pred, y_true, mask_zero=False):
        zeros = torch.zeros_like(y_pred[...,:1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if mask_zero:
            infs = zeros + 1e12
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)

        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
        aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
        neg_loss = all_loss + torch.log(aux_loss)

        return pos_loss + neg_loss

    def forward(self, logits, labels, mask_zero):
        """
        logits: [N, C, L, L]
        """
        shape = logits.shape
        labels = labels[..., 0] * shape[2] + labels[..., 1]
        logits = logits.reshape(shape[0], -1, np.prod(shape[2:]))
        
        return torch.mean(torch.sum(GPLinkerCrossEntropy.sparse_multilabel_categorical_crossentropy(logits, labels, mask_zero)))
