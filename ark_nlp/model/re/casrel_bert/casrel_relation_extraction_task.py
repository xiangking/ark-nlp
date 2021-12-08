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

import torch
import numpy as np

from torch.utils.data import DataLoader
from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask


def to_tup(triple_list):
    ret = []
    for triple in triple_list:
        ret.append(tuple([triple[0], triple[3], triple[4]]))
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


class CasRelRETask(SequenceClassificationTask):
    """
    基于CasRel Bert的联合关系抽取任务的Task
    
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
        return self.casrel_collate_fn(batch)

    def _evaluate_collate_fn(self, batch):
        return self.casrel_collate_fn(batch)

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
                'token_mapping': token_mapping}

    def fit(
        self,
        train_data=None,
        validation_data=None,
        lr=False,
        params=None,
        batch_size=32,
        epochs=1,
        gradient_accumulation_steps=1,
        **kwargs
    ):
        self.logs = dict()

        train_generator = self._on_train_begin(
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle=True,
            **kwargs
        )

        for epoch in range(epochs):

            train_data_prefetcher, inputs = self._on_epoch_begin(
                train_generator,
                **kwargs
            )

            step = 0

            while inputs is not None:

                self._on_step_begin(epoch, step, inputs, **kwargs)

                inputs = self._get_module_inputs_on_train(inputs, **kwargs)

                # forward
                outputs = self.module(**inputs)

                # 计算损失
                logits, loss = self._get_train_loss(inputs, outputs, **kwargs)

                loss = self._on_backward(inputs, outputs, logits, loss, **kwargs)

                # optimize
                if (step + 1) % gradient_accumulation_steps == 0:

                    # optimize
                    self._on_optimize(inputs, outputs, logits, loss, **kwargs)

                # setp evaluate
                self._on_step_end(step, inputs, outputs, logits, loss, **kwargs)

                step += 1

                inputs = train_data_prefetcher.next()

            self._on_epoch_end(epoch, **kwargs)

            if validation_data is not None:
                self.evaluate(validation_data, **kwargs)

    def _on_epoch_begin(self, train_generator, **kwargs):

        train_data_prefetcher = DataPreFetcher(
            train_generator,
            self.module.device
        )
        inputs = train_data_prefetcher.next()

        self.module.train()

        self._on_epoch_begin_record(**kwargs)

        return train_data_prefetcher, inputs

    def _get_module_inputs_on_train(
        self,
        inputs,
        **kwargs
    ):
        return inputs

    def _get_train_loss(
        self,
        inputs,
        outputs,
        **kwargs
    ):
        # 计算损失
        loss = self._compute_loss(inputs, outputs, **kwargs)

        self._compute_loss_record(**kwargs)

        return outputs, loss

    def _compute_loss(
        self,
        inputs,
        logits,
        verbose=True,
        **kwargs
    ):

        loss = self.loss_function(logits, inputs)

        return loss

    def evaluate(
        self,
        validation_data,
        evaluate_batch_size=1,
        h_bar=0.5,
        t_bar=0.5,
        **kwargs
    ):
        self.evaluate_logs = dict()

        evaluate_generator = self._on_evaluate_begin(
            validation_data,
            evaluate_batch_size,
            shuffle=False,
            **kwargs
        )

        test_data_prefetcher = DataPreFetcher(evaluate_generator, self.module.device)
        inputs = test_data_prefetcher.next()
        correct_num, predict_num, gold_num = 0, 0, 0
        step_ = 0

        with torch.no_grad():
            while inputs is not None:

                step_ += 1

                token_ids = inputs['input_ids']
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

                    pred_obj_heads, pred_obj_tails = self.module.get_objs_for_specific_sub(
                        sub_head_mapping,
                        sub_tail_mapping,
                        repeated_encoded_text
                    )
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

    def _on_evaluate_begin(
        self,
        validation_data,
        batch_size,
        shuffle,
        num_workers=0,
        **kwargs
    ):

        evaluate_generator = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._evaluate_collate_fn
        )

        self.module.eval()

        self._on_evaluate_begin_record(**kwargs)

        return evaluate_generator

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['correct_num'] = 0
        self.evaluate_logs['predict_num'] = 0
        self.evaluate_logs['gold_num'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_example'] = 0
