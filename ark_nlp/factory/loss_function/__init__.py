from torch.nn.modules.loss import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    SmoothL1Loss
)
from .casrel_loss import CasrelLoss
from .focal_loss import FocalLoss
from .label_smoothing_ce_loss import LabelSmoothingCrossEntropy
from .global_pointer_ce_loss import GlobalPointerCrossEntropy
from .r_drop_cross_entropy_loss import RDropCrossEntropyLoss


all_losses_dict = dict(
    binarycrossentropy=BCEWithLogitsLoss,
    bce=BCEWithLogitsLoss,
    crossentropy=CrossEntropyLoss,
    ce=CrossEntropyLoss,
    smoothl1=SmoothL1Loss,
    casrel=CasrelLoss,
    labelsmoothingcrossentropy=LabelSmoothingCrossEntropy,
    lsce=LabelSmoothingCrossEntropy,
    gpce=GlobalPointerCrossEntropy,
)


def get_loss(_loss):
    if isinstance(_loss, str):
        _loss = _loss.lower()
        _loss = _loss.replace('_', '')
        return all_losses_dict[_loss]()

    return _loss
