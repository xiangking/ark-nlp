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
# Author: Chenjie Shen, jimme.shen123@gmail.com.com
# Status: Active


import torch

from ark_nlp.factory.utils import conlleval
from ark_nlp.factory.task.base._token_classification import TokenClassificationTask
from torch.utils.data._utils.collate import default_collate


# TODO: 将该函数加入conlleval
def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


# TODO: 将该函数加入conlleval
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

        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])

        ent_r.extend(ent_set)
        ent_p.extend(predicts)
        for x in predicts:
            if x in ent_set:
                # ent_c += 1
                ent_c.append(x)
    return ent_c, ent_p, ent_r
    # return ent_c, ent_p, ent_r


class W2NERTask(TokenClassificationTask):
    """
    W2NER的命名实体识别Task

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

    def _train_collate_fn(self, batch):
        """将InputFeatures转换为Tensor"""

        input_ids = default_collate([f['input_ids'] for f in batch])
        attention_mask = default_collate([f['attention_mask'] for f in batch])
        token_type_ids = default_collate([f['token_type_ids'] for f in batch])
        grid_mask2d = default_collate([f['grid_mask2d'] for f in batch])
        dist_inputs = default_collate([f['dist_inputs'] for f in batch])
        pieces2word = default_collate([f['pieces2word'] for f in batch])
        label_ids = default_collate([f['label_ids'] for f in batch])
        input_lengths = default_collate([f['input_lengths'] for f in batch])
        entity_text = [f['entity_text'] for f in batch]

        tensors = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'grid_mask2d': grid_mask2d,
            'dist_inputs': dist_inputs,
            'pieces2word': pieces2word,
            'label_ids': label_ids,
            'input_lengths': input_lengths,
            'entity_text': entity_text,
        }

        return tensors

    def _evaluate_collate_fn(self, batch):
        return self._train_collate_fn(batch)

    def _compute_loss(
        self,
        inputs,
        logits,
        verbose=True,
        **kwargs
    ):
        active_loss = inputs['grid_mask2d'].view(-1) == 1
        active_logits = logits.reshape(-1, self.class_num)
        active_labels = torch.where(
            active_loss,
            inputs['label_ids'].view(-1),
            torch.tensor(self.loss_function.ignore_index).type_as(inputs['label_ids']
                                                                  )
        )
        loss = self.loss_function(active_logits, active_labels.long())

        return loss

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_example'] = 0

        self.evaluate_logs['labels'] = []
        self.evaluate_logs['logits'] = []
        self.evaluate_logs['input_lengths'] = []

        self.evaluate_logs['numerate'] = 0
        self.evaluate_logs['denominator'] = 0

        self.evaluate_logs['rights'] = []
        self.evaluate_logs['founds'] = []
        self.evaluate_logs['origins'] = []

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)

            logits = torch.argmax(logits, -1)
            entity_text, length = inputs['entity_text'], inputs['input_lengths']

            rights, founds, origins = decode(logits.cpu().numpy(), entity_text, length.cpu().numpy())

            self.evaluate_logs['rights'].extend(rights)
            self.evaluate_logs['founds'].extend(founds)
            self.evaluate_logs['origins'].extend(origins)

            self.evaluate_logs['eval_example'] += len(inputs['label_ids'])
            self.evaluate_logs['eval_step'] += 1
            self.evaluate_logs['eval_loss'] += loss.item()

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        id2cat=None,
        markup='bio',
        **kwargs
    ):

        if id2cat is None:
            id2cat = self.id2cat

        self.ner_metric = conlleval.SeqEntityScore(id2cat, markup=markup)

        self.ner_metric.rights = self.evaluate_logs['rights']
        self.ner_metric.founds = self.evaluate_logs['founds']
        self.ner_metric.origins = self.evaluate_logs['origins']

        print(len(self.ner_metric.rights), len(self.ner_metric.founds), len(self.ner_metric.origins))

        eval_info, entity_info = self.ner_metric.result()

        if is_evaluate_print:
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(
                self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step'],
                eval_info['acc'],
                eval_info['recall'],
                eval_info['f1'])
            )
