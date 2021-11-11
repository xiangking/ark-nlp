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
