import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class CasRelLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _compute_loss(gold, pred, mask):
        pred = pred.squeeze(-1)
        loss_ = F.binary_cross_entropy(pred, gold, reduction='none')
        if loss_.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        loss_ = torch.sum(loss_ * mask) / torch.sum(mask)
        return loss_

    def forward(self, logits, inputs):

        pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = logits

        sub_heads_loss = CasRelLoss._compute_loss(
            inputs['sub_heads'], pred_sub_heads,
            inputs['attention_mask']
        )
        sub_tails_loss = CasRelLoss._compute_loss(
            inputs['sub_tails'], pred_sub_tails,
            inputs['attention_mask']
        )
        obj_heads_loss = CasRelLoss._compute_loss(
            inputs['obj_heads'],
            pred_obj_heads,
            inputs['attention_mask']
        )
        obj_tails_loss = CasRelLoss._compute_loss(
            inputs['obj_tails'],
            pred_obj_tails,
            inputs['attention_mask']
        )

        loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)

        return loss
