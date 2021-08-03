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
from ark_nlp.factory.task.base._task import Task


class SequenceClassificationTask(Task):
    
    def __init__(self, *args, **kwargs):
        
        super(SequenceClassificationTask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'SequenceLevel'

    def _on_train_begin(
        self, 
        train_data, 
        validation_data, 
        batch_size,
        lr, 
        params, 
        shuffle,
        inputs_cols=None,
        **kwargs
    ):
        
        if self.class_num == None:
            self.class_num = train_data.class_num  
        
        if inputs_cols == None:
            self.inputs_cols = train_data.dataset_cols
        else:
            self.inputs_cols = inputs_cols
            
        train_generator = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
        self.train_generator_lenth = len(train_generator)
        
        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()
        
        self.module.train()
        
        self._on_train_begin_record(**kwargs)
        
        return train_generator
    
    def _on_train_begin_record(self, **kwargs):

        self.logs['global_step'] = 0 
        
        self.logs['tr_loss'] = 0
        self.logs['logging_loss'] = 0
            
    def _on_epoch_begin(self, **kwargs):
        
        self.module.train()
        
        self._on_epoch_begin_record(**kwargs)
            
    def _on_epoch_begin_record(self, **kwargs):

        self.logs['epoch_loss'] = 0
        self.logs['epoch_step'] = 0

        if verbose:
            self.logs['epoch_evaluation'] = 0
                
    def _on_step_begin(
        self, 
        epoch, 
        step, 
        inputs, 
        **kwargs
    ):
        
        self._on_step_begin_record(epoch, step, inputs, **kwargs)
            
    def _compute_loss(
        self, 
        inputs, 
        labels, 
        logits, 
        verbose=True,
        **kwargs
    ):  
        loss = self.loss_function(logits, labels)  
        
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
        
        if verbose:
            self.metric.update(inputs, labels, logits)
            self.logs['epoch_evaluation'] += self.metric.evaluate()
            self.metric.reset()
                
        self.logs['epoch_loss'] += loss.item() 
        self.logs['epoch_step'] += 1
        self.logs['global_step'] += 1
        
    def _on_backward(
        self, 
        inputs, 
        labels, 
        logits, 
        loss, 
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
        
        self._on_backward_record(**kwargs)
        
        return loss

    def _on_optimize(
        self, 
        step, 
        gradient_accumulation_step=1,
        **kwargs
    ):

        if (step + 1) % gradient_accumulation_step == 0:
            self.optimizer.step()  # 更新权值

            if self.ema_decay:
                self.ema.update(self.module.parameters())

            if self.scheduler:
                self.scheduler.step()  # 更新学习率
                
            self.optimizer.zero_grad()  # 清空梯度

            self._on_optimize_record(**kwargs)
                        
    def _on_step_end(
        self, 
        step,
        **kwargs
    ):
        self._on_step_end_record(step, **kwargs)

    def _on_step_end_record(
        self, 
        step,
        verbose=True,
        logging_step=100,
        **kwargs
    ):
        if verbose and (step + 1) % logging_step == 0:
            print('[{}/{}],train loss is:{:.6f},train evaluation is:{:.6f}'.format(
                step, 
                self.train_generator_lenth,
                self.logs['epoch_loss'] / self.logs['epoch_step'],
                self.logs['epoch_evaluation'] / self.logs['epoch_step']))
            
    def _on_epoch_end(
        self, 
        epoch,
        verbose=True,
        save_steps=0,
        save_module_path=None,
        **kwargs
    ):
        self._on_epoch_end_record(**kwargs)

    def _on_epoch_end_record(
        self, 
        epoch,
        verbose=True,
        **kwargs
    ):
        if verbose:
            print('epoch:[{}],train loss is:{:.6f},train evaluation is:{:.6f} \n'.format(
                epoch,
                self.logs['epoch_loss'] / self.logs['epoch_step'],
                self.logs['epoch_evaluation'] / self.logs['epoch_step']))  
            
    def _on_evaluate_begin(
        self, 
        validation_data, 
        batch_size, 
        shuffle, 
        **kwargs
    ):
        
        generator = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn)
        
        self.module.eval()
        
        self._on_evaluate_begin_record(**kwargs)
        
        return generator
    
    def _on_evaluate_begin_record(self, **kwargs):
        
        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_evaluation']  = 0

        self.evaluate_logs['eval_step']  = 0
        
        self.evaluate_logs['labels'] = []
        self.evaluate_logs['logits'] = []

    def _on_evaluate_epoch_begin(self, **kwargs):

        if self.ema_decay:
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())
        
        self._on_epoch_begin_record(**kwargs)
                    
    def _on_evaluate_step_end(self, inputs, labels, logits, loss, **kwargs):

        self.metric.update(inputs, labels, logits)

        self.evaluate_logs['nb_eval_examples'] +=  len(labels)
        self.evaluate_logs['nb_eval_steps']  += 1
        self.evaluate_logs['eval_loss'] += loss.item() * len(labels)
        self.evaluate_logs['eval_acc'] += torch.sum(preds == labels.data).item()
        
    def _on_evaluate_epoch_end(
        self, 
        **kwargs
    ):
        self._on_evaluate_epoch_end_record(**kwargs)

        if self.ema_decay:
            self.ema.restore(self.module.parameters())
    
    def _on_evaluate_end(
        self, 
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        **kwargs):
        
        labels_ = torch.cat(self.evaluate_logs['labels'], dim=0).cpu()
        preds_ = torch.argmax(torch.cat(self.evaluate_logs['logits'], dim=0), -1).cpu()

        
        f1_score = sklearn_metrics.f1_score(labels_, preds_, average='macro')
        report_ = sklearn_metrics.classification_report(labels_, preds_, target_names = [str(category_) for category_ in validation_data.categories])
        confusion_matrix_ = sklearn_metrics.confusion_matrix(labels_, preds_)


        if (save_steps > 0 and self.logs['global_step'] % save_steps == 0 or ):
            if save_module_path is None:
                prefix = './checkpoints/' + str(type(self.module.__class__.__name__)) + '_'
                save_module_path = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            torch.save(self.module.state_dict(), save_module_path) 



        
        if is_evaluate_print:
            print('classification_report: \n', report_)
            print('confusion_matrix_: \n', confusion_matrix_)
            print('test loss is:{:.6f},test acc is:{:.6f},f1_score is:{:.6f}'.format(self.evaluate_logs['eval_loss'] / self.evaluate_logs['nb_eval_steps'], 
                                                                                     self.evaluate_logs['eval_acc'] / self.evaluate_logs['nb_eval_examples'] ,
                                                                                     f1_score))

        del self.evaluate_logs
        gc.collect()

    def _get_module_inputs_on_train(
        self,
        inputs,
        labels,
        **kwargs
    ):
        return {col: inputs[col].to(self.device) for col in self.inputs_cols}

    def _get_module_label_on_train(
        self,
        inputs,
        **kwargs
    ):
        return inputs['label_ids'].to(self.device)

    def _get_module_inputs_on_eval(
        self,
        inputs,
        labels,
        **kwargs
    ):
        return {col: inputs[col].to(self.device) for col in self.inputs_cols}

    def _get_module_label_on_eval(
        self,
        inputs,
        **kwargs
    ):
        return inputs['label_ids'].to(self.device)

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

        self.id2cat = train_data.id2cat
        
        train_generator = self._on_train_begin(train_data, validation_data, batch_size, lr, params, shuffle=True, **kwargs)
                
        for epoch in range(epochs):
            
            self._on_epoch_begin(**kwargs)
            
            for step, inputs in enumerate(tqdm(train_generator)):
                
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
            
            self._on_epoch_end(epoch, **kwargs)
            
            if validation_data is not None:
                self.evaluate(validation_data, **kwargs)
    
    def evaluate(
        self, 
        validation_data, 
        evaluate_batch_size=16, 
        return_pred=False, 
        **kwargs
    ):
        self.evaluate_logs = dict()
        
        generator = self._on_evaluate_begin(validation_data, evaluate_batch_size, shuffle=False, **kwargs)
                
        with torch.no_grad():

            self._on_evaluate_epoch_begin(**kwargs)

            for step, inputs in enumerate(generator):
                
                labels = self._get_module_label_on_eval(inputs, **kwargs)
                inputs = self._get_module_inputs_on_eval(inputs, labels, **kwargs)
                
                # forward
                logits = self.module(**inputs)
                
                # compute loss
                loss = self._compute_loss(inputs, labels, logits, **kwargs)
                
                self._on_evaluate_step_end(inputs, labels, logits, loss, **kwargs)
            
            self._on_evaluate_epoch_end(**kwargs)

        self._on_evaluate_end(validation_data, **kwargs)
