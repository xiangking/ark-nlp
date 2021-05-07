"""
# Copyright 2021 Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

from ark_nlp.factory.task import TokenClassificationTask
from ark_nlp.factory.utils import conlleval


class CRFTask(TokenClassificationTask):
    def __init__(self, *args, **kwargs):
        super(CRFTask, self).__init__(*args, **kwargs)
    
    def _compute_loss(
        self, 
        inputs, 
        labels, 
        logits, 
        logs=None,
        verbose=True,
        **kwargs
    ):      
        loss = -1 * self.module.crf(emissions = logits, tags=labels, mask=inputs['attention_mask'])
        
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
        
    def _on_evaluate_step_end(self, inputs, labels, logits, loss, logs, **kwargs):
        
        tags = self.module.crf.decode(logits, inputs['attention_mask'])
        tags  = tags.squeeze(0)
        
        logs['labels'].append(labels)
        logs['logits'].append(tags)
        logs['input_lengths'].append(inputs['input_lengths'])
            
        logs['nb_eval_examples'] +=  len(labels)
        logs['nb_eval_steps']  += 1
        logs['eval_loss'] += loss.item()
        
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
        
        preds_ = torch.cat(logs['logits'], dim=0).cpu().numpy().tolist()        
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