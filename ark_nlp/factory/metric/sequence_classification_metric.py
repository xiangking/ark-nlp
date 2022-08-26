import torch
import numpy as np
import sklearn.metrics as sklearn_metrics

from torch import Tensor


class SequenceClassificationMetric(object):

    def __init__(self):
        self.labels = []
        self.preds = []

    def reset(self):
        self.labels = []
        self.preds = []

    def compute(self, preds, labels, categories=None):
        
        if categories:
            categories = [str(category) for category in categories]
        
        accuracy = float(np.sum(preds == labels) / len(labels))
        
        f1 = float(sklearn_metrics.f1_score(labels, preds, average='macro'))

        report = sklearn_metrics.classification_report(
            labels,
            preds,
            labels=range(categories) if categories else categories,
            target_names=categories)

        confusion_matrix = sklearn_metrics.confusion_matrix(labels, preds)

        return accuracy, f1, report, confusion_matrix

    def result(self, categories=None):
        labels = np.concatenate(self.preds)
        preds = np.concatenate(self.labels)
        
        accuracy, f1, report, confusion_matrix = self.compute(preds, labels, categories)
        
        return {'accuracy': accuracy, 'f1-score': f1, 'report': report, 'confusion-matrix': confusion_matrix}

    def update(self, preds, labels):
        if isinstance(preds, Tensor):
            preds = preds.numpy()

        if isinstance(labels, Tensor):
            labels = labels.numpy()

        self.preds.append(preds)
        self.labels.append(labels)