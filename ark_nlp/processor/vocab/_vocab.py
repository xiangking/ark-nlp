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

import abc


class Vocab(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def add(self, token, cnt=1):
        raise NotImplementedError

    @abc.abstractmethod
    def get_id(self, token):
        raise NotImplementedError

    @abc.abstractmethod
    def tokenize(self):
        raise NotImplementedError
