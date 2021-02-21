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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import tqdm
from tqdm import tqdm
import sklearn.metrics as sklearn_metrics

from ..loss_function import get_loss
from ..optimizer import get_optimizer
from ._task import Task



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
        
    def casrel_collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        batch.sort(key=lambda x: x[2], reverse=True)
        token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens = zip(*batch)
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
                'tokens': tokens}
        
    def _on_train_begin(
        self, 
        train_data, 
        validation_data, 
        batch_size,
        lr, 
        params, 
        logs,
        shuffle,
        num_workers=1,
        inputs_cols=None,
        **kwargs
    ):
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
        
        self._on_train_begin_record(logs, **kwargs)
        
        return train_generator
    
    def _on_train_begin_record(self, logs, **kwargs):
        
        logs['tr_loss'] = 0
        logs['logging_loss'] = 0
        
        return logs
    
    def _on_epoch_begin(self, logs, train_generator, **kwargs):
        
        train_data_prefetcher = DataPreFetcher(train_generator, self.module.device)
        inputs = train_data_prefetcher.next()
        
        self.module.train()
        
        self._on_epoch_begin_record(logs)
        
        return train_data_prefetcher, inputs
            
    def _on_epoch_begin_record(self, logs):
        
        logs['b_loss'] = 0
        logs['nb_tr_examples'] = 0
        logs['nb_tr_steps'] = 0
        
        return logs
        
    def _on_step_begin(
        self, 
        epoch, 
        step, 
        inputs, 
        logs, 
        **kwargs
    ):
        self._on_step_begin_record(epoch, step, inputs, logs, **kwargs)
        pass
    
    def _on_step_begin_record(
        self, 
        epoch, 
        step, 
        inputs, 
        logs, 
        **kwargs
    ):
        pass
            
    def _compute_loss(
        self, 
        inputs, 
        labels, 
        logits, 
        logs=None,
        verbose=True,
        **kwargs
    ):  
        
        loss = self.loss_function(logits, inputs)     
        
        if logs:
            self._compute_loss_record(inputs, labels, logits, loss, logs, verbose, **kwargs)
                
        return loss
    
    def _compute_loss_record(
        self,
        inputs, 
        labels, 
        logits, 
        loss, 
        logs,
        verbose,
        **kwargs
    ):        
        logs['nb_tr_examples'] += inputs['input_ids'].size(0)
                
        logs['b_loss'] += loss.item()
        logs['nb_tr_steps'] += 1
        
        return logs

    def _on_backward(
        self, 
        inputs, 
        labels, 
        logits, 
        loss, 
        logs,
        **kwargs
    ):
                
        loss.backward() 
        
        self._on_backward_record(logs)
        
        return loss
    
    def _on_optimize(self, step, logs, **kwargs):
        self.optimizer.step()  # 更新权值
        if self.scheduler:
            self.scheduler.step()  # 更新学习率
                
        self.optimizer.zero_grad()  # 清空梯度
        
        self._on_optimize_record(logs)
        
        return step
    
    def _on_step_end(
        self, 
        step,
        logs,
        verbose=True,
        print_step=100,
        **kwargs
    ):
        if verbose and (step + 1) % print_step == 0:
            print('[{}/{}],train loss is:{:.6f}'.format(
                step, 
                self.train_generator_lenth,
                logs['b_loss'] / logs['nb_tr_steps']))
            
        self._on_step_end_record(logs)
            
    def _on_epoch_end(
        self, 
        epoch,
        logs,
        verbose=True,
        **kwargs
    ):
        if verbose:
            print('epoch:[{}],train loss is:{:.6f} \n'.format(
                epoch,
                logs['b_loss'] / logs['nb_tr_steps']))  
            
        self._on_epoch_end_record(logs)
            
    def _on_evaluate_begin(
        self, 
        validation_data, 
        batch_size, 
        logs, 
        shuffle, 
        **kwargs
    ):
        
        generator = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
        
        self.module.eval()
        
        self._on_evaluate_begin_record(logs, **kwargs)
        
        return generator
    
    def _on_evaluate_begin_record(self, logs, **kwargs):
        
        logs['eval_loss'] = 0
        logs['nb_eval_steps']  = 0
        logs['nb_eval_examples']  = 0
        
        return logs     
                    
    def _on_evaluate_step_end(self, inputs, labels, logits, loss, logs, **kwargs):
        
        _, preds = torch.max(logits, 1)

        logs['nb_eval_examples'] +=  inputs['input_ids'].size(0)
        logs['nb_eval_steps']  += 1
        logs['eval_loss'] += loss.item() * inputs['input_ids'].size(0)
        
        return logs
    
    def _on_evaluate_end(
        self, 
        validation_data,
        logs,
        epoch=1,
        is_evaluate_print=True,
        **kwargs):
        
        
        
        if is_evaluate_print:
            print('classification_report: \n', report_)
            print('confusion_matrix_: \n', confusion_matrix_)
            print('test loss is:{:.6f}'.format(logs['eval_loss'] / logs['nb_eval_steps']))

    def fit(
        self, 
        train_data=None, 
        validation_data=None, 
        lr=1e-3,
        params=None,
        batch_size=32,
        epochs=1,
        **kwargs
    ):
        logs = dict()
        
        train_generator = self._on_train_begin(train_data, validation_data, batch_size, lr, params, logs, shuffle=True, **kwargs)
                
        for epoch in range(epochs):
            
            train_data_prefetcher, inputs = self._on_epoch_begin(logs, train_generator, **kwargs)
            
            step = 0
            
            while inputs is not None:
                
                self._on_step_begin(epoch, step, inputs, logs, **kwargs)
                                                
                labels = inputs['label_ids']
                                
                # forward
                logits = self.module(**inputs)
                                
                # 计算损失
                loss = self._compute_loss(inputs, labels, logits, logs, **kwargs)
                                                
                # loss backword
                loss = self._on_backward(inputs, labels, logits, loss, logs, **kwargs)
                
                # optimize
                step = self._on_optimize(step, logs, **kwargs)
                
                # setp evaluate
                self._on_step_end(step, logs, **kwargs)
                
                step += 1
            
            self._on_epoch_end(epoch, logs, **kwargs)
            
            if validation_data is not None:
                self.evaluate(validation_data, **kwargs)

    def predict(
        self, 
        test_data, 
        batch_size=16, 
        shuffle=False, 
        is_proba=False
    ):
        
        preds = []
        probas=[]
        
        self.module.eval()
        generator = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        for step, inputs in enumerate(generator):
            inputs = {col: inputs[col].to(self.device) for col in self.inputs_cols}
            logits = self.module(**inputs)

            preds.extend(torch.max(logits, 1)[1].cpu().numpy())  
            if is_proba:
                probas.extend(F.softmax(logits, 1).cpu().detach().numpy())  

        if is_prob:
            return preds, probas
        
        return preds
    
    def predict_proba(
        self, 
        test_data, 
        batch_size=16, 
        shuffle=False
    ):
        return self.predict(test_data, batch_size, shuffle, is_proba=True)[-1]
    
    def evaluate(
        self, 
        validation_data, 
        evaluate_batch_size=16, 
        return_pred=False, 
        **kwargs
    ):
        logs = dict()
        
        generator = self._on_evaluate_begin(validation_data, evaluate_batch_size, logs, shuffle=False, **kwargs)
                
        with torch.no_grad():
                        
            for step, inputs in enumerate(generator):
                
                labels = inputs['label_ids'].to(self.device)
                inputs = {col: inputs[col].to(self.device) for col in self.inputs_cols}
                
                # forward
                logits = self.module(**inputs)
                
                # compute loss
                loss = self._compute_loss(inputs, labels, logits, **kwargs)
                
                self._on_evaluate_step_end(inputs, labels, logits, loss, logs, **kwargs)
                
        self._on_evaluate_end(validation_data, logs)