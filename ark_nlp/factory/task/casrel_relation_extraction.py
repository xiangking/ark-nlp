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

import tqdm
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics as sklearn_metrics

from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.task import Task


def to_tup(triple_list):
    ret = []
    for triple in triple_list:
        ret.append(tuple(triple))
    return ret

class DataPreFetcher(object):
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


class CasrelRETask(Task):

    def __init__(self, *args, **kwargs):

        super(CasrelRETask, self).__init__(*args, **kwargs)

        warnings.warn("The CasrelRETask is deprecated, please use other CasrelRETask ( from ark_nlp.model.re.casrel_bert.casrel_relation_extraction_task import CasRelRETask )", DeprecationWarning)

    def casrel_collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        batch.sort(key=lambda x: x[2], reverse=True)
        token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens, token_mapping = zip(*batch)
        cur_batch = len(batch)
        max_text_len = max(text_len)
        batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
        batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
        batch_sub_heads = torch.Tensor(cur_batch, max_text_len).zero_()
        batch_sub_tails = torch.Tensor(cur_batch, max_text_len).zero_()
        batch_sub_head = torch.Tensor(cur_batch, max_text_len).zero_()
        batch_sub_tail = torch.Tensor(cur_batch, max_text_len).zero_()
        batch_obj_heads = torch.Tensor(cur_batch, max_text_len, self.class_num).zero_()
        batch_obj_tails = torch.Tensor(cur_batch, max_text_len, self.class_num).zero_()

        for i in range(cur_batch):
            batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
            batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
            batch_sub_heads[i, :text_len[i]].copy_(torch.from_numpy(sub_heads[i]))
            batch_sub_tails[i, :text_len[i]].copy_(torch.from_numpy(sub_tails[i]))
            batch_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sub_head[i]))
            batch_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sub_tail[i]))
            batch_obj_heads[i, :text_len[i], :].copy_(torch.from_numpy(obj_heads[i]))
            batch_obj_tails[i, :text_len[i], :].copy_(torch.from_numpy(obj_tails[i]))

        return {'input_ids': batch_token_ids,
                'attention_mask': batch_masks,
                'sub_heads': batch_sub_heads,
                'sub_tails': batch_sub_tails,
                'sub_head': batch_sub_head,
                'sub_tail': batch_sub_tail,
                'obj_heads': batch_obj_heads,
                'obj_tails': batch_obj_tails,
                'label_ids': triples,
                'tokens': tokens,
                'token_mapping': token_mapping
               }

    def _on_train_begin(
        self, 
        train_data, 
        validation_data, 
        batch_size,
        lr, 
        params, 
        shuffle,
        num_workers=1,
        inputs_cols=None,
        **kwargs
    ):
        self.id2cat = train_data.id2cat

        if self.class_num == None:
            self.class_num = train_data.class_num  

        if inputs_cols == None:
            self.inputs_cols = train_data.dataset_cols

        train_generator = DataLoader(dataset=train_data,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     pin_memory=True,
                                     num_workers=num_workers,
                                     collate_fn=self.casrel_collate_fn)  

        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()

        self.module.train()

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_train_begin_record(self, **kwargs):

        self.logs['tr_loss'] = 0
        self.logs['logging_loss'] = 0

    def _on_epoch_begin(self, train_generator, **kwargs):

        train_data_prefetcher = DataPreFetcher(train_generator, self.module.device)
        inputs = train_data_prefetcher.next()

        self.module.train()

        self._on_epoch_begin_record(**kwargs)

        return train_data_prefetcher, inputs

    def _on_epoch_begin_record(self, **kwargs):

        self.logs['epoch_loss'] = 0
        self.logs['epoch_example'] = 0
        self.logs['epoch_step'] = 0

    def _compute_loss(
        self, 
        inputs, 
        labels, 
        logits, 
        verbose=True,
        **kwargs
    ):  

        loss = self.loss_function(logits, inputs)

        if self.logs:
            self._compute_loss_record(inputs, labels, logits, loss, verbose, **kwargs)

        return loss

    def _compute_loss_record(
        self,
        inputs, 
        labels, 
        logits, 
        loss, 
        verbose,
        **kwargs
    ):
        self.logs['epoch_example'] += len(labels)

        self.logs['epoch_loss'] += loss.item()
        self.logs['epoch_step'] += 1

    def _on_backward(
        self, 
        inputs, 
        labels, 
        logits, 
        loss, 
        **kwargs
    ):

        loss.backward()

        self._on_backward_record(**kwargs)

        return loss

    def _on_optimize(self, step, **kwargs):
        self.optimizer.step()  # 更新权值
        if self.scheduler:
            self.scheduler.step()  # 更新学习率

        self.optimizer.zero_grad()  # 清空梯度

        self._on_optimize_record(**kwargs)

        return step

    def _on_step_end(
        self, 
        step,
        verbose=True,
        print_step=100,
        **kwargs
    ):
        if verbose and (step + 1) % print_step == 0:
            print('[{}/{}],train loss is:{:.6f}'.format(
                step, 
                self.train_generator_lenth,
                self.logs['epoch_loss'] / self.logs['epoch_step']))

        self._on_step_end_record(**kwargs)

    def _on_epoch_end(
        self, 
        epoch,
        verbose=True,
        **kwargs
    ):
        if verbose:
            print('epoch:[{}],train loss is:{:.6f} \n'.format(
                epoch,
                self.logs['epoch_loss'] / self.logs['epoch_step']))  

        self._on_epoch_end_record(**kwargs)

    def _get_module_inputs_on_train(
        self,
        inputs,
        labels,
        **kwargs
    ):
        return inputs

    def _get_module_label_on_train(
        self,
        inputs,
        **kwargs
    ):
        return inputs['label_ids']
        # return inputs['label_ids'].to(self.device)

    def fit(
        self, 
        train_data=None, 
        validation_data=None, 
        lr=False,
        params=None,
        batch_size=32,
        epochs=1,
        **kwargs
    ):
        self.logs = dict()

        train_generator = self._on_train_begin(train_data, validation_data, batch_size, lr, params, shuffle=True, **kwargs)

        for epoch in range(epochs):

            train_data_prefetcher, inputs = self._on_epoch_begin(train_generator, **kwargs)

            step = 0

            while inputs is not None:

                self._on_step_begin(epoch, step, inputs, **kwargs)

                labels = self._get_module_label_on_train(inputs, **kwargs)
                inputs = self._get_module_inputs_on_train(inputs, labels, **kwargs)

                # forward
                logits = self.module(**inputs)

                # 计算损失
                loss = self._compute_loss(inputs, labels, logits, **kwargs)

                # loss backword
                loss = self._on_backward(inputs, labels, logits, loss, **kwargs)

                # optimize
                step = self._on_optimize(step, **kwargs)

                # setp evaluate
                self._on_step_end(step, **kwargs)

                step += 1

                inputs = train_data_prefetcher.next()

            self._on_epoch_end(epoch, **kwargs)

            if validation_data is not None:
                self.evaluate(validation_data, **kwargs)

    def _on_evaluate_begin(
        self, 
        validation_data, 
        batch_size, 
        shuffle, 
        num_workers=1,
        **kwargs
    ):

        test_data_loader = DataLoader(dataset=validation_data,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1,
                                      collate_fn=self.casrel_collate_fn)  

        self.module.eval()

        self._on_evaluate_begin_record(**kwargs)

        return test_data_loader

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['correct_num'] = 0
        self.evaluate_logs['predict_num'] = 0
        self.evaluate_logs['gold_num'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_example'] = 0

    def evaluate(
        self, 
        validation_data, 
        evaluate_batch_size=1, 
        return_pred=False, 
        h_bar=0.5,
        t_bar=0.5,
        **kwargs
    ):
        self.evaluate_logs = dict()

        generator = self._on_evaluate_begin(validation_data, evaluate_batch_size, shuffle=False, **kwargs)

        test_data_prefetcher = DataPreFetcher(generator, self.module.device)
        inputs = test_data_prefetcher.next()
        correct_num, predict_num, gold_num = 0, 0, 0
        step_ = 0

        with torch.no_grad():
            while inputs is not None:

                step_ += 1

                token_ids = inputs['input_ids']
                tokens = inputs['tokens'][0]
                token_mapping = inputs['token_mapping'][0]
                mask = inputs['attention_mask']

                encoded_text = self.module.bert(token_ids, mask)[0]

                pred_sub_heads, pred_sub_tails = self.module.get_subs(encoded_text)
                sub_heads, sub_tails = np.where(pred_sub_heads.cpu()[0] > h_bar)[0], np.where(pred_sub_tails.cpu()[0] > t_bar)[0]

                subjects = []
                for sub_head in sub_heads:
                    sub_tail = sub_tails[sub_tails >= sub_head]
                    if len(sub_tail) > 0:
                        sub_tail = sub_tail[0]
                        subject = ''.join([token_mapping[index_] if index_ < len(token_mapping) else '' for index_ in range(sub_head-1, sub_tail)])
                        if subject == '':
                            continue
                        subjects.append((subject, sub_head, sub_tail))

                if subjects:
                    triple_list = []
                    repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                    sub_head_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                    sub_tail_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                    for subject_idx, subject in enumerate(subjects):
                        sub_head_mapping[subject_idx][0][subject[1]] = 1
                        sub_tail_mapping[subject_idx][0][subject[2]] = 1
                    sub_tail_mapping = sub_tail_mapping.to(repeated_encoded_text)
                    sub_head_mapping = sub_head_mapping.to(repeated_encoded_text)

                    pred_obj_heads, pred_obj_tails = self.module.get_objs_for_specific_sub(sub_head_mapping, 
                                                                                          sub_tail_mapping, 
                                                                                          repeated_encoded_text)
                    for subject_idx, subject in enumerate(subjects):
                        sub = subject[0]

                        obj_heads, obj_tails = np.where(pred_obj_heads.cpu()[subject_idx] > h_bar), np.where(pred_obj_tails.cpu()[subject_idx] > t_bar)
                        for obj_head, rel_head in zip(*obj_heads):
                            for obj_tail, rel_tail in zip(*obj_tails):
                                if obj_head <= obj_tail and rel_head == rel_tail:
                                    rel = self.id2cat[int(rel_head)]
                                    obj = ''.join([token_mapping[index_] if index_ < len(token_mapping) else '' for index_ in range(obj_head-1, obj_tail)])
                                    triple_list.append((sub, rel, obj))
                                    break
                    triple_set = set()
                    for s, r, o in triple_list:
                        if o == '' or s == '':
                            continue
                        triple_set.add((s, r, o))
                    pred_list = list(triple_set)
                else:
                    pred_list = []

                pred_triples = set(pred_list)

                gold_triples = set(to_tup(inputs['label_ids'][0]))

                correct_num += len(pred_triples & gold_triples)

                if step_ < 11:
                    print('pred_triples: ', pred_triples)
                    print('gold_triples: ', gold_triples)

                predict_num += len(pred_triples)
                gold_num += len(gold_triples)

                inputs = test_data_prefetcher.next()

        print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))

        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        print("precision: {}, recall: {}, f1_score: {}".format(precision, recall, f1_score))

        return precision, recall, f1_score