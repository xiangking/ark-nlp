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
# Author: Chenjie Shen, jimme.shen123@gmail.com
# Status: Active

import torch
from torch.utils.data._utils.collate import default_collate

from ark_nlp.factory.metric import W2NERSpanMetric
from ark_nlp.factory.task.base._token_classification import TokenClassificationTask


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def decode(outputs, entities, length):
    ent_r, ent_p, ent_c = [], [], []
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):
        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(l):
            for j in range(i + 1, l):
                if instance[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        for head in head_dict:
            find_entity(head, [], head_dict[head])

        predicts = set(
            [convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])

        ent_r.extend(ent_set)
        ent_p.extend(predicts)

    return ent_r, ent_p


class W2NERTask(TokenClassificationTask):
    """
    W2NER的Task

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

    def __init__(self, *args, **kwargs):

        super(W2NERTask, self).__init__(*args, **kwargs)

        if 'metric' not in kwargs:
            self.metric = W2NERSpanMetric()

    def _train_collate_fn(self, batch):
        """将InputFeatures转换为Tensor"""

        input_ids = default_collate([f['input_ids'] for f in batch])
        attention_mask = default_collate([f['attention_mask'] for f in batch])
        token_type_ids = default_collate([f['token_type_ids'] for f in batch])
        grid_mask2d = default_collate([f['grid_mask2d'] for f in batch])
        dist_inputs = default_collate([f['dist_inputs'] for f in batch])
        pieces2word = default_collate([f['pieces2word'] for f in batch])
        label_ids = default_collate([f['label_ids'] for f in batch])
        sequence_length = default_collate([f['sequence_length'] for f in batch])
        entity_text = [f['entity_text'] for f in batch]

        tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'grid_mask2d': grid_mask2d,
            'dist_inputs': dist_inputs,
            'pieces2word': pieces2word,
            'label_ids': label_ids,
            'sequence_length': sequence_length,
            'entity_text': entity_text,
        }

        return tensors

    def _evaluate_collate_fn(self, batch):
        return self._train_collate_fn(batch)

    def compute_loss(self, inputs, logits, **kwargs):
        active_loss = inputs['grid_mask2d'].view(-1) == 1
        active_logits = logits.reshape(-1, self.class_num)
        active_labels = torch.where(
            active_loss, inputs['label_ids'].view(-1),
            torch.tensor(self.loss_function.ignore_index).type_as(inputs['label_ids']))
        loss = self.loss_function(active_logits, active_labels.long())

        return loss

    def on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():

            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['loss'] += loss.item()

            logits = torch.argmax(logits, -1)
            origins, founds = decode(logits.cpu().numpy(),
                                     inputs['entity_text'],
                                     inputs['sequence_length'].cpu().numpy())

            self.metric.update(origins, founds)
