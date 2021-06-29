from torch.nn.modules.loss import *
from .casrel_loss import CasrelLoss
from .focal_loss import FocalLoss
from .label_smoothing_ce_loss import LabelSmoothingCrossEntropy
from .global_pointer_ce_loss import GlobalPointerCrossEntropy


all_losses_dict = dict(binarycrossentropy=BCEWithLogitsLoss,
                       bce=BCEWithLogitsLoss,
                       crossentropy=CrossEntropyLoss,
                       ce=CrossEntropyLoss,
                       smoothl1=SmoothL1Loss,
                       casrel=CasrelLoss,
                       labelsmoothingcrossentropy=LabelSmoothingCrossEntropy,
                       lsce=LabelSmoothingCrossEntropy,
                       gpce=GlobalPointerCrossEntropy,
                      )


def get_loss(loss_):
    if isinstance(loss_, str):
        loss_ = loss_.lower()
        loss_ = loss_.replace('_', '')
        return all_losses_dict[loss_]()

    return loss_