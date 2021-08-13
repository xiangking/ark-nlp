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

import re
import tqdm
import torch
import collections
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
from torch._six import string_classes

from ark_nlp.factory.utils import conlleval
from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.metric import topk_accuracy
from ark_nlp.factory.task.base._task import Task
from ark_nlp.factory.task.base._token_classification import TokenClassificationTask

from ark_nlp.factory.utils import conlleval
from ark_nlp.factory.metric import BiaffineSpanMetrics
from ark_nlp.factory.metric import SpanMetrics


class BIONERTask(TokenClassificationTask):
    
    def __init__(self, *args, **kwargs):
        
        super(BIONERTask, self).__init__(*args, **kwargs)

    def _on_evaluate_epoch_end(
        self, 
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        id2cat=None,
        markup='bio',
        **kwargs
    ):

        if id2cat == None:
            id2cat = self.id2cat
        
        self.ner_metric = conlleval.SeqEntityScore(id2cat, markup=markup)
        preds_ = torch.argmax(torch.cat(self.evaluate_logs['logits'], dim=0), -1).numpy().tolist()        
        labels_ = torch.cat(self.evaluate_logs['labels'], dim=0).numpy().tolist()
        input_lens_ = torch.cat(self.evaluate_logs['input_lengths'], dim=0).numpy()
                
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
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step'], 
                                                                                              eval_info['acc'], 
                                                                                              eval_info['recall'],
                                                                                              eval_info['f1']))    


class CRFNERTask(TokenClassificationTask):
    
    def _compute_loss(
        self, 
        inputs, 
        logits, 
        verbose=True,
        **kwargs
    ):      
        loss = -1 * self.module.crf(emissions = logits, tags=inputs['label_ids'], mask=inputs['attention_mask'])
        
        self._compute_loss_record(inputs, logits, loss, verbose, **kwargs)
                
        return loss

    def _on_evaluate_step_end(self, inputs, logits, **kwargs):

        with torch.no_grad():
            # compute loss
            loss = self._compute_loss(inputs, logits, **kwargs)
            
            tags = self.module.crf.decode(logits, inputs['attention_mask'])
            tags  = tags.squeeze(0)
        
        self.logs['labels'].append(inputs['label_ids'].cpu())
        self.logs['logits'].append(tags.cpu())
        self.logs['input_lengths'].append(inputs['input_lengths'].cpu())
            
        self.logs['eval_example'] +=  len(inputs['label_ids'])
        self.logs['eval_step']  += 1
        self.logs['eval_loss'] += loss.item() 
                
    def _on_evaluate_epoch_end(
        self, 
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        id2cat=None,
        markup='bio',
        **kwargs
    ):

        if id2cat == None:
            id2cat = self.id2cat
        
        self.ner_metric = conlleval.SeqEntityScore(id2cat, markup=markup)
        
        preds_ = torch.cat(self.evaluate_logs['logits'], dim=0).numpy().tolist()        
        labels_ = torch.cat(self.evaluate_logs['labels'], dim=0).numpy().tolist()
        input_lens_ = torch.cat(self.evaluate_logs['input_lengths'], dim=0).numpy()
                
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
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step'], 
                                                                                              eval_info['acc'], 
                                                                                              eval_info['recall'],
                                                                                              eval_info['f1']))  


class BiaffineNERTask(TokenClassificationTask):
    
    def _compute_loss(
        self, 
        inputs, 
        logits, 
        verbose=True,
        **kwargs
    ):      
        
        span_label = inputs['label_ids'].view(size=(-1,))
        span_logits = logits.view(size=(-1, self.class_num))
        
        span_loss = self.loss_function(span_logits, span_label)
    
        span_mask = inputs['span_mask'].view(size=(-1,))
        
        span_loss *= span_mask
        loss = torch.sum(span_loss) / inputs['span_mask'].size()[0]
        
        if self.logs:
            self._compute_loss_record(inputs, logits, loss, verbose, **kwargs)
                
        return loss
        
    def _on_evaluate_step_end(self, inputs, logits, **kwargs):

        with torch.no_grad():
            # compute loss
            loss = self._compute_loss(inputs, logits, **kwargs)
            
            logits = torch.nn.functional.softmax(logits, dim=-1)
                
        self.evaluate_logs['labels'].append(inputs['label_ids'].cpu())
        self.evaluate_logs['logits'].append(logits.cpu())
            
        self.evaluate_logs['eval_example'] += len(inputs['label_ids'])
        self.evaluate_logs['eval_step']  += 1
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

        if id2cat == None:
            id2cat = self.id2cat
            
        biaffine_metric = BiaffineSpanMetrics()

        preds_ = torch.cat(self.evaluate_logs['logits'], dim=0)     
        labels_ = torch.cat(self.evaluate_logs['labels'], dim=0)

        with torch.no_grad():
            recall, precise, span_f1 = biaffine_metric(preds_, labels_)

        if is_evaluate_print:
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step'], 
                                                                                              precise, 
                                                                                              recall,
                                                                                              span_f1)) 


class GlobalPointerNERTask(TokenClassificationTask):
    def __init__(self, *args, **kwargs):
        super(GlobalPointerNERTask, self).__init__(*args, **kwargs)
        
    def _compute_loss(
        self, 
        inputs, 
        logits, 
        verbose=True,
        **kwargs
    ):      
        loss = self.loss_function(logits, inputs['label_ids'])
        
        if self.logs:
            self._compute_loss_record(inputs, logits, loss, verbose, **kwargs)
                
        return loss

    def _compute_loss_record(
        self,
        inputs, 
        logits, 
        loss, 
        verbose,
        **kwargs
    ):        
        self.logs['epoch_loss'] += loss.item() 
        self.logs['epoch_step'] += 1
            
    def _on_evaluate_begin_record(self, **kwargs):
        
        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_step']  = 0
        self.evaluate_logs['eval_example']  = 0
        
        self.evaluate_logs['labels'] = []
        self.evaluate_logs['logits'] = []
        self.evaluate_logs['input_lengths'] = []
        
        self.evaluate_logs['numerate'] = 0
        self.evaluate_logs['denominator'] = 0
                
    def _on_evaluate_step_end(self, inputs, logits, **kwargs):
        
        with torch.no_grad():

            # compute loss
            loss = self._compute_loss(inputs, logits, **kwargs)

            numerate, denominator = conlleval.global_pointer_f1_score(inputs['label_ids'].cpu(), logits.cpu())   
            self.evaluate_logs['numerate'] += numerate
            self.evaluate_logs['denominator'] += denominator
            
        self.evaluate_logs['eval_example'] +=  len(inputs['label_ids'])
        self.evaluate_logs['eval_step']  += 1
        self.evaluate_logs['eval_loss'] += loss.item()
        
    def _on_evaluate_epoch_end(
        self, 
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        id2cat=None,
        **kwargs):

        if id2cat == None:
            id2cat = self.id2cat

        if is_evaluate_print:
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step'], 
                                                                                              self.evaluate_logs['numerate'], 
                                                                                              self.evaluate_logs['denominator'],
                                                                                              2*self.evaluate_logs['numerate']/self.evaluate_logs['denominator'])) 


class SpanNERTask(TokenClassificationTask):
    
    def _compute_loss(
        self, 
        inputs, 
        logits, 
        verbose=True,
        **kwargs
    ):
        start_logits = logits[0]
        end_logits = logits[1]  

        start_logits = start_logits.view(-1, len(self.id2cat))
        end_logits = end_logits.view(-1, len(self.id2cat))
    
        active_loss = inputs['attention_mask'].view(-1) == 1
    
        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]
                
        active_start_labels = inputs['start_label_ids'].long().view(-1)[active_loss]
        active_end_labels = inputs['end_label_ids'].long().view(-1)[active_loss]
        
        start_loss = self.loss_function(active_start_logits, active_start_labels)
        end_loss = self.loss_function(active_end_logits, active_end_labels)
        
        loss = start_loss + end_loss
                
        if self.logs:
            self._compute_loss_record(inputs, logits, loss, verbose, **kwargs)
                
        return loss
    
    def _on_evaluate_epoch_begin(self, **kwargs):
        
        self.metric = SpanMetrics(self.id2cat)

        if self.ema_decay:
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())
        
        self._on_epoch_begin_record(**kwargs)

    def _on_evaluate_step_end(self, inputs, logits, **kwargs):

        with torch.no_grad():
            # compute loss
            loss = self._compute_loss(inputs, logits, **kwargs)
        
        length = inputs['attention_mask'].cpu().numpy().sum() - 2
        
        S = []
        start_logits = logits[0]
        end_logits = logits[1]
        
        start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:length+1]
        end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:length+1]
        
        for i, s_l in enumerate(start_pred):
            if s_l == 0:
                continue
            for j, e_l in enumerate(end_pred[i:]):
                if s_l == e_l:
                    S.append((s_l, i, i + j))
                    break
                    
        self.metric.update(true_subject=inputs['label_ids'][0], pred_subject=S)
            
        self.evaluate_logs['eval_example'] +=  len(inputs['label_ids'])
        self.evaluate_logs['eval_step']  += 1
        self.evaluate_logs['eval_loss'] += loss.item()
        
    def _on_evaluate_epoch_end(
        self, 
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        id2cat=None,
        **kwargs):

        if id2cat == None:
            id2cat = self.id2cat
        
        with torch.no_grad():
            eval_info, entity_info = self.metric.result()

        if is_evaluate_print:
            print('eval_info: ', eval_info)
            print('entity_info: ', entity_info)
            
    def _collate_fn(self, batch):
                
        def default_collate(batch):
            r"""Puts each data field into a tensor with outer dimension batch size"""

            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                return torch.stack(batch, 0, out=out)
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                    and elem_type.__name__ != 'string_':
                if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                    # array of string classes and object
                    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                    return default_collate([torch.as_tensor(b) for b in batch])
                elif elem.shape == ():  # scalars
                    return torch.as_tensor(batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float64)
            elif isinstance(elem, int):
                return torch.tensor(batch)
            elif isinstance(elem, string_classes):
                return batch
            elif isinstance(elem, collections.abc.Mapping):
                dict_ = {}
                for key in elem:
                    if key != 'label_ids':
                        try:
                            dict_[key] = default_collate([d[key] for d in batch])
                        except:
                            dict_[key] = default_collate([torch.as_tensor(d[key]) for d in batch])
                    else:
                        dict_[key] = [d[key] for d in batch]
                return dict_                
                # return {key: default_collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
                return elem_type(*(default_collate(samples) for samples in zip(*batch)))
            elif isinstance(elem, collections.abc.Sequence):
                # check to make sure that the elements in batch have consistent size
                it = iter(batch)
                elem_size = len(next(it))
                if not all(len(elem) == elem_size for elem in it):
                    raise RuntimeError('each element in list of batch should be of equal size')
                transposed = zip(*batch)
                return [default_collate(samples) for samples in transposed]

            raise TypeError(default_collate_err_msg_format.format(elem_type))
            
        return default_collate(batch)