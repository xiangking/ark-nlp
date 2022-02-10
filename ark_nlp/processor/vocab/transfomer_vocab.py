# Copyright (c) 2020 DataArk Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xiang Wang, xiangking1995@163.com
# Status: Active


import jieba

from typing import List
from transformers import BertTokenizer


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
