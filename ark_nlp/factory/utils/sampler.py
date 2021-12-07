import torch
import torch.utils.data


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    从给定的不平衡数据集中采用torch.multinomial进行随机采样，暂时不可用于类似命名实体之类一个样本有多个标签的情况

    Args:
        dataset (:obj:`ark_nlp dataset`): batch文本
        indices (:obj:`list` or :obj:`None`, optional, defaults to None): 给定的数据集索引列表，默认为None，由dataset处生成
        num_samples (:obj:`int` or :obj:`None`, optional, defaults to None): 需要的indices长度，默认为None，使用完整的indices长度
        callback_get_label (:obj:`function`, optional, defaults to None): 自定义获取样本label的函数

    Examples::

        >>> train_generator = DataLoader(train_data, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_data))

    Reference:
        [1]  https://github.com/ufoym/imbalanced-dataset-sampler
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            return dataset[idx]['label_ids']

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
