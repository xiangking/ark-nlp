"""
# Copyright Xiang Wang, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
# http://www.apache.org/licenses/LICENSE-2.0

Author: Xiang Wang, xiangking1995@163.com
Status: Active
"""

from ark_nlp.dataset import PairMergeSentenceClassificationDataset


class TMDataset(PairMergeSentenceClassificationDataset):
    def __init__(self, *args, **kwargs):

        super(TMDataset, self).__init__(*args, **kwargs)
