# Copyright (c) 2021 DataArk Authors. All Rights Reserved.
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


from ark_nlp.nn.base.bert import Bert
from ark_nlp.nn.base.bert import BertForSequenceClassification
from ark_nlp.nn.base.bert import BertForTokenClassification


class Ernie(Bert):
    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(Ernie, self).__init__(config, encoder_trained)


class ErnieForSequenceClassification(BertForSequenceClassification):
    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(ErnieForSequenceClassification, self).__init__(config, encoder_trained)


class ErnieForTokenClassification(BertForTokenClassification):
    def __init__(
        self,
        config,
        encoder_trained=True
    ):
        super(BertForTokenClassification, self).__init__(config, encoder_trained)
