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

from ark_nlp.processor.vocab._vocab import Vocab


class LabelVocab(Vocab):

    def __init__(self, initial_labels=None):
        self.id2label = {}
        self.label2id = {}

        self.initial_labels = initial_labels
        for label in self.initial_labels:
            self.add(label)

    def add(self, label, cnt=1):
        if label in self.label2id:
            idx = self.label2id[label]
        else:
            idx = len(self.id2label)
            self.id2label[idx] = label
            self.label2id[label] = idx
        return idx

    def convert_to_ids(self, labels):
        ids = [self.get_id(label) for label in labels]
        return ids

    def recover_from_ids(self, ids, stop_id=None):
        labels = []
        for i in ids:
            labels += [self.get_token(i)]
        return labels

    def recover_id2label(self):
        id2label_temp = {}
        for label_iter, idx_iter in self.label2id:
            id2label_temp[idx_iter] = label_iter
        return id2label_temp

    def get_id(self, label):
        try:
            return self.label2id[label]
        except KeyError:
            raise Exception("Invalid label!")

    def get_label(self, idx):
        try:
            return self.id2label[idx]
        except KeyError:
            raise Exception("Invalid id!")
