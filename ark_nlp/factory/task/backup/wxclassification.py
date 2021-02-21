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
from ..metric import topk
from ._task import Task


class ClassificationTask(Task):
    
    def __init__(self, *args, **kwargs):
        
        super(ClassificationTask, self).__init__(*args, **kwargs)
        
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
            
        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()
            
        self.module.train()
        
        logs['global_step'] = 0
        logs['tr_loss'] = 0
        logs['logging_loss'] = 0
        logs['train_generator_lenth'] = len(train_generator)
        
        return train_generator, logs
    
    def _on_epoch_begin(self, logs, **kwargs):
        
        self.module.train()
        
        logs['b_loss'] = 0
        logs['b_acc'] = 0
        logs['nb_tr_examples'] = 0
        logs['nb_tr_steps'] = 0
                    
        return logs
            
    def _compute_loss(
        self, 
        inputs, 
        labels, 
        logits, 
        **kwargs
    ):
        loss = self.loss_function(logits, labels)
                
        return loss
    
    def _on_train_record(
        inputs, 
        labels, 
        logits, 
        loss, 
        logs,
        verbose=True,
        **kwargs
    ):
        logs['nb_tr_examples'] += inputs['input_ids'].size(0)
        
        if verbose:
            with torch.no_grad():
                _, preds = torch.max(logits, 1)
                logs['b_acc'] += torch.sum(preds == labels).item()
                
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
        verbose=True,
        gradient_accumulation_steps=1,
        grad_clip=None,
        **kwargs
    ):
        
        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            
        loss.backward() 
        
        if grad_clip != None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), grad_clip)
        
        return loss, logs
    
    def _on_optimize(self, step, logs, gradient_accumulation_steps=1, **kwargs):
        if (step + 1) % gradient_accumulation_steps == 0:
            self.optimizer.step()  # 更新权值
            if self.scheduler:
                self.scheduler.step()  # 更新学习率
                
            self.optimizer.zero_grad()  # 清空梯度
            
            logs['global_step'] += 1
            
        return step, logs
    
    def _on_train_step_metric(
        self, 
        logs, 
        step, 
        verbose=True,
        print_step=100,
        **kwargs
    ):
        if verbose and (step + 1) % print_step == 0:
            print('[{}/{}],train loss is:{:.6f},train acc is:{:.6f}'.format(
                step, 
                logs['train_generator_lenth'],
                logs['b_loss'] / logs['nb_tr_steps'],
                logs['b_acc'] / logs['nb_tr_examples']))
            
    def _on_train_epoch_metric(
        self, 
        logs,
        epoch,
        verbose=True,
        **kwargs
    ):
        if verbose:
            print('epoch:[{}],train loss is:{:.6f},train acc is:{:.6f} \n'.format(
                epoch,
                logs['b_loss'] / logs['nb_tr_steps'],
                logs['b_acc'] / logs['nb_tr_examples']))   
            
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
        
        logs['eval_loss'] = 0
        logs['eval_acc']  = 0
        logs['nb_eval_steps']  = 0
        logs['nb_eval_examples']  = 0
        
        logs['eval_top2_acc']  = 0
        logs['eval_top3_acc']  = 0
        logs['eval_top4_acc']  = 0
        logs['eval_top5_acc']  = 0
        
        logs['labels'] = []
        logs['logits'] = []
        
        return generator, logs        
                    
    def _on_evaluate_step_metric(self, inputs, labels, logits, loss, logs, **kwargs):
        
        _, preds = torch.max(logits, 1)
        
        logs['labels'].append(labels)
        logs['logits'].append(logits)
            
        logs['nb_eval_examples'] +=  inputs['input_ids'].size(0)
        logs['nb_eval_steps']  += 1
        logs['eval_loss'] += loss.item() * inputs['input_ids'].size(0)
        logs['eval_acc'] += torch.sum(preds == labels.data).item()
        
        logs['eval_top2_acc']  += topk(logits, labels, 2, reduction='sum')
        logs['eval_top3_acc']  += topk(logits, labels, 3, reduction='sum')
        logs['eval_top4_acc']  += topk(logits, labels, 4, reduction='sum')
        logs['eval_top5_acc']  += topk(logits, labels, 5, reduction='sum')
        
        return logs
    
    def _on_evaluate_epoch_metric(
        self, 
        validation_data,
        logs,
        epoch=1,
        is_evaluate_print=True,
        **kwargs):
        
        labels_ = torch.cat(logs['labels'], dim=0).cpu()
        preds_ = torch.argmax(torch.cat(logs['logits'], dim=0), -1).cpu()

        
        f1_score = sklearn_metrics.f1_score(labels_, preds_, average='macro')
        report_ = sklearn_metrics.classification_report(labels_, preds_, target_names = validation_data.categories)
        confusion_matrix_ = sklearn_metrics.confusion_matrix(labels_, preds_)
        
        if is_evaluate_print:
            print('classification_report: \n', report_)
            print('confusion_matrix_: \n', confusion_matrix_)
            print('test loss is:{:.6f},test acc is:{:.6f},f1_score is:{:.6f}'.format(logs['eval_loss'] / logs['nb_eval_steps'], 
                                                                                     logs['eval_acc'] / logs['nb_eval_examples'] ,
                                                                                     f1_score))
            print('top2 acc is:{:.6f}, top3 acc is:{:.6f}, top4 acc is:{:.6f}, top5 acc is:{:.6f}'.format(logs['eval_top2_acc'] / logs['nb_eval_examples'],
                                                                                                          logs['eval_top3_acc'] / logs['nb_eval_examples'],
                                                                                                          logs['eval_top4_acc'] / logs['nb_eval_examples'],
                                                                                                          logs['eval_top5_acc'] / logs['nb_eval_examples']))
    
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
        
        train_generator, logs = self._on_train_begin(train_data, validation_data, batch_size, lr, params, logs, shuffle=True, **kwargs)
                
        for epoch in range(epochs):
            
            logs = self._on_epoch_begin(logs, **kwargs)
            
            for step, inputs in enumerate(tqdm(train_generator)):
                
                labels = inputs['label_ids'].to(self.device)
                inputs = {col: inputs[col].to(self.device) for col in self.inputs_cols}
                
                # forward
                logits = self.module(**inputs)
                                
                # 计算损失
                loss = self._compute_loss(inputs, labels, logits, **kwargs)
                
                logs = self._on_train_record(inputs, labels, logits, loss, logs, **kwargs)
                                
                # loss backword
                loss, logs = self._on_backward(inputs, labels, logits, loss, logs, **kwargs)
                
                # optimize
                step, logs = self._on_optimize(step, logs, **kwargs)
                
                # setp evaluate
                self._on_train_step_metric(logs, step, **kwargs)
            
            self._on_train_epoch_metric(logs, epoch, **kwargs)
            
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
        
        generator, logs = self._on_evaluate_begin(validation_data, evaluate_batch_size, logs, shuffle=False, **kwargs)
                
        with torch.no_grad():
            for step, inputs in enumerate(generator):
                labels = inputs['label_ids'].to(self.device)
                inputs = {col: inputs[col].to(self.device) for col in self.inputs_cols}
                
                # forward
                logits = self.module(**inputs)
                
                # compute loss
                loss = self._compute_loss(inputs, labels, logits, **kwargs)
                
                logs = self._on_evaluate_step_metric(inputs, labels, logits, loss, logs, **kwargs)
                
        self._on_evaluate_epoch_metric(validation_data, logs)
                