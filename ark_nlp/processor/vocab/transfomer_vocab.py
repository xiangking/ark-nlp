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

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from zhon.hanzi import punctuation
from functools import lru_cache
from transformers import BertTokenizer
from transformers.tokenization_utils import _is_control
from ark_nlp.processor.vocab._vocab import Vocab


class TransfomerWithBlankVocab(BertTokenizer):
    def tokenize(self, text, **kwargs) -> List[str]:
        tokens = []
        for span_ in text.split():
            tokens += self._tokenize(span_)
            tokens += [' ']
        return tokens[:-1]


class RoFormerVocab(BertTokenizer):
    
    def tokenize(self, text):
        return list(jieba.cut(text))