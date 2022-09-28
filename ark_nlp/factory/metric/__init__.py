import torch

from torch import nn
from collections import Counter
from ark_nlp.factory.metric.span_metric import SpanMetric
from ark_nlp.factory.metric.span_metric import W2NERSpanMetric
from ark_nlp.factory.metric.triple_metric import TripleMetric
from ark_nlp.factory.metric.biaffine_span_metric import BiaffineSpanMetric
from ark_nlp.factory.metric.global_pointer_metric import GlobalPointerMetric
from ark_nlp.factory.metric.sequence_classification_metric import SequenceClassificationMetric
from ark_nlp.factory.metric.spearman_correlation_metric import SpearmanCorrelationMetric
from ark_nlp.factory.metric.relation_extraction_metric import RelationExtractionMetric


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
        logits (torch.FloatTensor): 模型预测的概率值
        labels (torch.LongTensor): 真实的标签值
        k (int): Top K
        ignore_index (int, optional): 忽略的索引值, 默认值为: -100
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
