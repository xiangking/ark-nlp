import torch

from torch import nn
from collections import Counter


def topk_accuracy(
    logits,
    labels,
    k,
    ignore_index=-100,
    reduction='mean'
):
    """
    计算 TopK Accuracy

    Args:
        logits (:obj:`torch.FloatTensor`): 模型预测的概率值
        labels (:obj:`torch.LongTensor`): 真实的标签值
        k (:obj:`int`): Top K
        ignore_index (:obj:`int`, optional, defaults to -100):
        reduction (:obj:`str`, optional, defaults to "mean"): acc汇聚方式

    :Returns:
        TopK Accuracy

    """

    topk_pred = logits.topk(k, dim=1)[1]
    weights = (labels != ignore_index).float()
    num_labels = weights.sum()
    topk_acc = (labels.unsqueeze(1) == topk_pred).any(1).float() * weights

    if reduction in ['mean', 'sum']:
        topk_acc = topk_acc.sum()

    if reduction == 'mean':
        topk_acc = topk_acc / num_labels

    return topk_acc


class BiaffineSpanMetrics(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        logits = torch.argmax(logits, dim=-1)
        batch_size, seq_len, hidden = labels.shape
        logits = logits.view(batch_size, seq_len, hidden)

        logits = logits.view(size=(-1,)).float()
        labels = labels.view(size=(-1,)).float()

        ones = torch.ones_like(logits)
        zero = torch.zeros_like(logits)
        y_pred = torch.where(logits < 1, zero, ones)

        ones = torch.ones_like(labels)
        zero = torch.zeros_like(labels)
        y_true = torch.where(labels < 1, zero, ones)

        corr = torch.eq(logits, labels).float()
        corr = torch.mul(corr, y_true)
        recall = torch.sum(corr) / (torch.sum(y_true) + 1e-8)
        precision = torch.sum(corr) / (torch.sum(y_pred) + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)

        return recall, precision, f1


class SpanMetrics(object):

    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])
