import torch
from torch import nn

def topk_accuracy(logits, labels, k, *, ignore_index=-100, reduction='mean'):
    """
    计算 TopK Accuracy
    
    :param logits: (Tensor) 模型预测的概率值
    :param labels: (Tensor) 真实的标签值
    :param k: (Int) Top K
    
    :returns:  
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
        batch_size, seq_len,hidden = labels.shape
        logits = logits.view(batch_size, seq_len,hidden)

        logits = logits.view(size=(-1,)).float()
        labels = labels.view(size=(-1,)).float()

        ones = torch.ones_like(logits)
        zero = torch.zeros_like(logits)
        y_pred = torch.where(logits<1, zero, ones)

        ones = torch.ones_like(labels)
        zero = torch.zeros_like(labels)
        y_true = torch.where(labels<1, zero, ones)

        corr = torch.eq(logits, labels).float()
        corr = torch.mul(corr,y_true)
        recall = torch.sum(corr) / (torch.sum(y_true) + 1e-8)
        precision = torch.sum(corr) / (torch.sum(y_pred) + 1e-8)
        f1=2*recall*precision/(recall+precision+1e-8)

        return recall, precision, f1