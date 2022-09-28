import torch.nn as nn
import torch.nn.functional as F


class PFNLoss(nn.Module):
    def __init__(
        self,
    ):
        super(PFNLoss, self).__init__()
        self.loss_ner = nn.BCELoss(reduction='sum')
        self.loss_re_head = nn.BCELoss(reduction='sum')
        self.loss_re_tail = nn.BCELoss(reduction='sum')

    def forward(self, ner_pred, ner_label, re_pred_head, re_pred_tail, re_label_head, re_label_tail):
        seq_len = ner_pred.size(1)
        ner_loss = self.loss_ner(ner_pred, ner_label) / seq_len
        re_head_loss = self.loss_re_head(re_pred_head, re_label_head) / seq_len
        re_tail_loss = self.loss_re_tail(re_pred_tail, re_label_tail) / seq_len
        loss = ner_loss + re_head_loss + re_tail_loss

        return loss