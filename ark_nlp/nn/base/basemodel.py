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
import torch
import time


class BasicModule(torch.nn.Module):
    """
    封装了nn.Module，主要是提供了save和load两个方法

    Attributes:
        model_name (str): 模型名称
    """  # noqa: ignore flake8"

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path):
        """可加载指定路径的模型"""
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """保存模型，默认使用“模型名字+时间”作为文件名"""
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name


class Flat(torch.nn.Module):
    """Flat类，把输入reshape成（batch_size,dim_length）"""

    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
