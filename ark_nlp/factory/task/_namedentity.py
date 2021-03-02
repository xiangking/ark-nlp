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
from ..utils import conlleval

class NamedentityTask(Task):
    
    def __init__(self, *args, **kwargs):
        
        super(NamedentityTask, self).__init__(*args, **kwargs)
        
    def _on_train_begin(
        self, 
        train_data, 
        validation_data, 
        batch_size,
        lr, 
        params, 
        logs,
        shuffle,
        inputs_cols=None,
        **kwargs
    ):
        if self.class_num == None:
            self.class_num = train_data.class_num  
        
        if inputs_cols == None:
            self.inputs_cols = train_data.dataset_cols
            
        train_generator = DataLoader(train_data, batch_size=batch_size, shuffle=True)
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
    
    def _on_epoch_begin(self, logs, **kwargs):
        
        self.module.train()
        
        self._on_epoch_begin_record(logs)
            
    def _on_epoch_begin_record(self, logs):
        
        logs['b_loss'] = 0
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
        
        
        active_loss = inputs['attention_mask'].view(-1) == 1
        active_logits = logits.view(-1, self.class_num)
        active_labels = torch.where(active_loss, 
                                    labels.view(-1), 
                                    torch.tensor(self.loss_function.ignore_index).type_as(labels)
                                   )
        loss = self.loss_function(active_logits, active_labels)
        
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
            print('epoch:[{}],train loss is:{:.6f}\n'.format(
                epoch,
                logs['b_loss'] / logs['nb_tr_steps']))  
            
        self._on_epoch_end_record(logs)

    def _get_module_inputs_on_train(
        self,
        inputs
        label,
        **kwargs
    ):
        return {col: inputs[col].to(self.device) for col in self.inputs_cols}

    def _get_module_label_on_train(
        self,
        inputs
        **kwargs
    ):
        return inputs['label_ids'].to(self.device)

    def _get_module_inputs_on_eval(
        self,
        inputs,
        label,
        **kwargs
    ):
        return {col: inputs[col].to(self.device) for col in self.inputs_cols}

    def _get_module_label_on_eval(
        self,
        inputs,
        label,
        **kwargs
    ):
        return {col: inputs[col].to(self.device) for col in self.inputs_cols}

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
        
        self.id2cat = train_data.id2cat
        
        train_generator = self._on_train_begin(train_data, validation_data, batch_size, lr, params, logs, shuffle=True, **kwargs)
                
        for epoch in range(epochs):
            
            self._on_epoch_begin(logs, **kwargs)
            
            for step, inputs in enumerate(tqdm(train_generator)):
                
                self._on_step_begin(epoch, step, inputs, logs, **kwargs)
                                
                labels = self._get_module_inputs_on_train(inputs, **kwargs)
                inputs = self._get_module_inputs_on_train(inputs, label, **kwargs)
                                
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
        
        logs['labels'] = []
        logs['logits'] = []
        logs['input_lengths'] = []
        
        return logs     
                    
    def _on_evaluate_step_end(self, inputs, labels, logits, loss, logs, **kwargs):
        
        logs['labels'].append(labels)
        logs['logits'].append(logits)
        logs['input_lengths'].append(inputs['input_lengths'])
            
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
        id2cat=None,
        markup='bio',
        **kwargs):

        if id2cat == None:
            id2cat = self.id2cat
        
        self.ner_metric = conlleval.SeqEntityScore(id2cat, markup=markup)
        preds_ = torch.argmax(torch.cat(logs['logits'], dim=0), -1).cpu().numpy().tolist()        
        labels_ = torch.cat(logs['labels'], dim=0).cpu().numpy().tolist()
        input_lens_ = torch.cat(logs['input_lengths'], dim=0).cpu().numpy()
                
        for index_, label_ in enumerate(labels_):
            label_list_ = []
            pred_list_ = []
            for jndex_, _ in enumerate(label_):
                if jndex_ == 0:
                    continue
                elif jndex_ == input_lens_[index_]-1:
                    self.ner_metric.update(pred_paths=[pred_list_], label_paths=[label_list_])
                    break
                else:
                    label_list_.append(labels_[index_][jndex_])
                    pred_list_.append(preds_[index_][jndex_])        
        
        eval_info, entity_info = self.ner_metric.result()

        if is_evaluate_print:
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(logs['eval_loss'] / logs['nb_eval_steps'], 
                                                                                              eval_info['acc'], 
                                                                                              eval_info['recall'],
                                                                                              eval_info['f1']))    
    
    def evaluate(
        self, 
        validation_data, 
        evaluate_batch_size=16, 
        return_pred=False, 
        **kwargs
    ):
        logs = dict()
        self.logs = logs
        
        generator = self._on_evaluate_begin(validation_data, evaluate_batch_size, logs, shuffle=False, **kwargs)
                
        with torch.no_grad():
                        
            for step, inputs in enumerate(generator):
                
                labels = self._get_module_label_on_eval(inputs, **kwargs)
                inputs = self._get_module_inputs_on_eval(inputs, labels, **kwargs)
                
                # forward
                logits = self.module(**inputs)
                
                # compute loss
                loss = self._compute_loss(inputs, labels, logits, **kwargs)
                
                self._on_evaluate_step_end(inputs, labels, logits, loss, logs, **kwargs)
                
        self._on_evaluate_end(validation_data, logs)