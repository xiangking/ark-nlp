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

import torch
import numpy as np

from collections import defaultdict
from ark_nlp.factory.metric import TripleMetric
from ark_nlp.factory.task.base import SequenceClassificationTask


def to_tup(triple_list):
    ret = []
    for triple in triple_list:
        ret.append(tuple([triple[0], triple[3], triple[4]]))
    return ret


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """
    Numpy函数，将序列padding到同一长度

    Reference:
        [1] https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py
    """
    
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


class GPLinkerRETask(SequenceClassificationTask):
    """
    基于GPLinker的联合关系抽取任务的Task
    
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

        super(GPLinkerRETask, self).__init__(*args, **kwargs)

        if 'metric' not in kwargs:
            self.metric = TripleMetric()

    def _train_collate_fn(self, batch):
        return self.gplinker_collate_fn(batch)

    def _evaluate_collate_fn(self, batch):
        return self.gplinker_collate_fn(batch)

    def gplinker_collate_fn(self, batch):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        batch_triples = []
        batch_token_mapping = []
        for item in batch:
            input_ids, attention_mask, token_type_ids, entity_labels, head_labels, tail_labels, triples, token_mapping = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_triples.append(triples)
            batch_token_mapping.append(token_mapping)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()
        
        return {
            'input_ids': batch_token_ids,
            'attention_mask': batch_mask_ids,
            'token_type_ids': batch_token_type_ids,
            'entity_label_ids': batch_entity_labels,
            'head_label_ids': batch_head_labels,
            'tail_label_ids': batch_tail_labels,
            'label_ids': batch_triples,
            'token_mapping': batch_token_mapping
        }

    def get_module_inputs_on_train(self, inputs, **kwargs):
        """模型输入处理方法"""
        for col in inputs.keys():
            if type(inputs[col]) is torch.Tensor:
                inputs[col] = inputs[col].to(self.device)

        return inputs

    def get_module_inputs_on_evaluate(self, inputs, **kwargs):
        for col in inputs.keys():
            if type(inputs[col]) is torch.Tensor:
                inputs[col] = inputs[col].to(self.device)

        return inputs

    def get_train_loss(self, inputs, outputs, **kwargs):
        # 计算损失
        loss = self.compute_loss(inputs, outputs, **kwargs)

        return outputs, loss

    def compute_loss(self, inputs, logits, **kwargs):
        
        entity_logits, head_logits, tail_logits = logits

        entity_loss = self.loss_function(entity_logits, inputs['entity_label_ids'], mask_zero=True)
        head_loss = self.loss_function(head_logits, inputs['head_label_ids'], mask_zero=True)
        tail_loss = self.loss_function(tail_logits, inputs['tail_label_ids'], mask_zero=True)
        
        loss = (entity_loss + head_loss + tail_loss) / 3

        return loss
    
    def on_evaluate_step_end(self, inputs, outputs, threshold=0.0, **kwargs):

        batch_size, _ = inputs['input_ids'].size()
        token_mapping_list =  inputs['token_mapping']

        entity_logits, head_logits, tail_logits = [output.cpu().numpy() for output in outputs]
        
        for idx in range(batch_size):
        
            subjects, objects = set(), set()
            outputs[0][:, [0, -1]] -= np.inf
            outputs[0][:, :, [0, -1]] -= np.inf
            
            for entity_type, head_idx, tail_idx in zip(*np.where(entity_logits[idx] > 0)):
                if entity_type == 0:
                    subjects.add((head_idx, tail_idx))
                else:
                    objects.add((head_idx, tail_idx))
        
            pred_triples = set()
            for subject_head_idx, subject_tail_idx in subjects:
                for object_head_idx, object_tail_idx in objects:
                    head_relation_set = np.where(head_logits[idx][:, subject_head_idx, object_head_idx] > threshold)[0]
                    tail_relation_set = np.where(tail_logits[idx][:, subject_tail_idx, object_tail_idx] > threshold)[0]
                    relation_set = set(head_relation_set) & set(tail_relation_set)
                    for relation_type in relation_set:
                        subject = ''.join([token_mapping_list[idx][index_] if index_ < len(token_mapping_list[idx]) else '' for index_ in range(subject_head_idx-1, subject_tail_idx)])
                        obj = ''.join([token_mapping_list[idx][index_] if index_ < len(token_mapping_list[idx]) else '' for index_ in range(object_head_idx-1, object_tail_idx)])
                        pred_triples.add((
                            subject, 
                            self.id2cat[relation_type],
                            obj
                        ))
 
            gold_triples = set(to_tup(inputs['label_ids'][idx]))
        
            if self.metric:
                self.metric.update(preds=pred_triples, labels=gold_triples)
                
    @property
    def default_loss_function(self):
        return 'gplinker'
