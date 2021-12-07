"""
# Copyright 2020 Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

import dgl
import torch

from ark_nlp.factory.task import SequenceClassificationTask


class TextLevelGCNTask(SequenceClassificationTask):
    """
    TextLevelGCN文本分类任务的Task
    
    Args:
        module: 深度学习模型
        optimizer: 训练模型使用的优化器名或者优化器对象
        loss_function: 训练模型使用的损失函数名或损失函数对象
        class_num (:obj:`int` or :obj:`None`, optional, defaults to None): 标签数目
        scheduler (:obj:`class`, optional, defaults to None): scheduler对象
        n_gpu (:obj:`int`, optional, defaults to 1): GPU数目
        device (:obj:`class`, optional, defaults to None): torch.device对象，当device为None时，会自动检测是否有GPU
        cuda_device (:obj:`int`, optional, defaults to 0): GPU编号，当device为None时，根据cuda_device设置device
        ema_decay (:obj:`int` or :obj:`None`, optional, defaults to None): EMA的加权系数
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):

        super(TextLevelGCNTask, self).__init__(*args, **kwargs)

    def _train_collate_fn(
        self,
        batch
    ):
        batch_graph = []
        batch_label_ids = []

        for sample in batch:
            sample_graph = sample['sub_graph'].to(self.device)
            sample_graph.ndata['h'] = self.module.node_embed(torch.Tensor(sample['node_ids']).type(torch.long).to(self.device))
            sample_graph.edata['w'] = self.module.edge_embed(torch.Tensor(sample['edge_ids']).type(torch.long).to(self.device))

            batch_graph.append(sample_graph)
            batch_label_ids.append(sample['label_ids'])

        batch_graph = dgl.batch(batch_graph)

        return {'sub_graph': batch_graph, 'label_ids': torch.Tensor(batch_label_ids).type(torch.long)}

    def _evaluate_collate_fn(self, batch):
        return self._train_collate_fn(batch)
