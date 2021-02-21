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
import abc
import pickle
import jieba
import unicodedata

from zhon.hanzi import punctuation
from functools import lru_cache
from transformers import BertTokenizer
from transformers.tokenization_utils import _is_control
from .base_vocab import Vocab


class TransfomerVocab(BertTokenizer):
    
    def _tokenize(self, text):
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or _is_control(ch):
                continue
            elif '\u4e00' <= ch <= '\u9fff':
                spaced += ch
            elif ch == ' ':
                spaced += ' ' + '[unused1]' + ' '
            else:
                spaced += ' ' + ch + ' '
        tokens = []
        for word in re.split(' ', spaced.strip()):
            tokens += self.wordpiece_tokenizer.tokenize(word)
        return tokens