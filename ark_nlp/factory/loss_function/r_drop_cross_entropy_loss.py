import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from torch.nn.modules.loss import _WeightedLoss


class RDropCrossEntropyLoss(_WeightedLoss):
    r"""
    """
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = 'mean',
        rdrop_alpha: float = 1.0
    ) -> None:
        super(RDropCrossEntropyLoss, self).__init__(
            weight,
            size_average,
            reduce,
            reduction
        )
        self.ignore_index = ignore_index
        self.rdrop_alpha = rdrop_alpha

    def compute_kl_loss(self, p, q):

        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1),
            F.softmax(q, dim=-1),
            reduction='none'
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1),
            F.softmax(p, dim=-1),
            reduction='none'
        )

        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        return (p_loss + q_loss) / 2

    def forward(
        self,
        input_a: Tensor,
        input_b: Tensor,
        target: Tensor
    ) -> Tensor:

        ce_loss_a = F.cross_entropy(
            input_a,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )

        ce_loss_b = F.cross_entropy(
            input_b,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )

        ce_loss = 0.5 * (ce_loss_a + ce_loss_b)

        kl_loss = self.compute_kl_loss(input_a, input_b)

        return ce_loss + self.rdrop_alpha * kl_loss
