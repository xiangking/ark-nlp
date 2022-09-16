# Copyright (c) 2020 DataArk Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xiang Wang, xiangking1995@163.com
# Status: Active


import dgl
import torch

from ark_nlp.factory.task import SequenceClassificationTask


class TextLevelGCNTask(SequenceClassificationTask):
    """
    TextLevelGCN文本分类任务的Task
    
    Args:
        module: 深度学习模型
        optimizer (str or torch.optim.Optimizer or None, optional): 训练模型使用的优化器名或者优化器对象, 默认值为: None
        loss_function (str or object or None, optional): 训练模型使用的损失函数名或损失函数对象, 默认值为: None
        scheduler (torch.optim.lr_scheduler.LambdaLR, optional): scheduler对象, 默认值为: None
        tokenizer (object or None, optional): 分词器, 默认值为: None
        class_num (int or None, optional): 标签数目, 默认值为: None
        gpu_num (int, optional): GPU数目, 默认值为: 1
        device (torch.device, optional): torch.device对象, 当device为None时, 会自动检测是否有GPU
        cuda_device (int, optional): GPU编号, 当device为None时, 根据cuda_device设置device, 默认值为: 0
        ema_decay (int or None, optional): EMA的加权系数, 默认值为: None
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

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
