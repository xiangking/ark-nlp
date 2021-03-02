import torch.nn as nn

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