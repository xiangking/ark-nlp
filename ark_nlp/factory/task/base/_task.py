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


import time
import torch

from torch.optim import Optimizer
from torch.utils.data._utils.collate import default_collate
from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.lr_scheduler import get_scheduler
from ark_nlp.factory.utils.ema import EMA


class Task(object):
    """
    所有Task类的基类，封装Task类通用的方法和属性

    Args:
        module: 深度学习模型
        optimizer: 训练模型使用的优化器名或者优化器对象
        loss_function: 训练模型使用的损失函数名或损失函数对象
        tokenizer (:obj:`class` or :obj:`None`, optional, defaults to None): 分词器
        class_num (:obj:`int` or :obj:`None`, optional, defaults to None): 标签数目
        scheduler (:obj:`class`, optional, defaults to None): scheduler对象
        n_gpu (:obj:`int`, optional, defaults to 1): GPU数目
        device (:obj:`class`, optional, defaults to None): torch.device对象，当device为None时，会自动检测是否有GPU
        cuda_device (:obj:`int`, optional, defaults to 0): GPU编号，当device为None时，根据cuda_device设置device
        ema_decay (:obj:`int` or :obj:`None`, optional, defaults to None): EMA的加权系数
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

    def __init__(
        self,
        module,
        optimizer,
        loss_function,
        tokenizer=None,
        class_num=None,
        scheduler=None,
        n_gpu=1,
        device=None,
        cuda_device=0,
        ema_decay=None,
        **kwargs
    ):
        self.fit_counter = 0
        self.module = module
        self.tokenizer = tokenizer

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.set_loss_function(loss_function)

        self.class_num = class_num

        self.n_gpu = n_gpu
        self.device = device

        if self.device is None:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                self.device = "cpu"

        self.module.to(self.device)

        if self.n_gpu > 1:
            self.module = torch.nn.DataParallel(self.module)

        self.ema_decay = ema_decay
        if self.ema_decay:
            self.ema = EMA(self.module.parameters(), decay=self.ema_decay)

    def set_loss_function(
        self,
        loss_function
    ):
        if loss_function is None:
            self.loss_function = get_loss(self.default_loss_function)
        elif isinstance(loss_function, str) or isinstance(loss_function, object):
            self.loss_function = get_loss(self.default_loss_function)
        else:
            raise ValueError("The loss function type does not exist")

        return self.loss_function

    def set_optimizer(
        self,
        lr=None,
        eps=None,
        weight_decay=None,
        params=None
    ):
        # 通过params对optimizer内的参数进行修改
        if isinstance(self.optimizer, Optimizer) and not callable(self.optimizer) and params is not None:
            for index, param_group in enumerate(self.optimizer.param_groups):
                for key in (set(self.optimizer.param_groups[index].keys()) - set(params[index].keys())):
                    params[index][key] = self.optimizer.param_groups[index][key]
            self.optimizer.param_groups = params

        # 当optimizer还未被创建时，该部分代码负责创建optimizer
        if params is None:
            params = [{"params": [p for p in self.optimizer.parameters() if p.requires_grad]}]
        if self.optimizer is None:
            self.optimizer = get_optimizer(self.default_optimizer, params)
        if isinstance(self.optimizer, str) or callable(self.optimizer):
            self.optimizer = get_optimizer(self.optimizer, params)
        # 经过上述判断条件后仍然未创建optimizer, 则抛出相关创建异常
        if not isinstance(self.optimizer, Optimizer):
            raise ValueError("The optimizer type does not exist")

        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        if eps is not None:
            for param_group in self.optimizer.param_groups:
                param_group['eps'] = eps

        if weight_decay is not None:
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = weight_decay

    def set_scheduler(
        self,
        epochs,
        batch_size,
        **kwargs
    ):
        if self.scheduler is not None:
            warmup_step = self.train_generator_lenth * epochs
            self.scheduler = get_scheduler(
                self.scheduler,
                self.optimizer,
                self.train_generator_lenth,
                warmup_step,
                **kwargs
            )

    def save(self, save_path: str, save_mode: str = 'pth'):
        """
        提供多种方式保存模型
        
        Args:
            save_path: 保存的路径
            save_mode (string, optional):
                保存的方式
                "huggingface"表示会以transformers库保存预训练模型的格式进行保存
                "pth"表示module会以torch.save的方式保存模型权重
                默认值为: "pth"
        """  # noqa: ignore flake8"
        if save_mode == 'huggingface':
            self.module.save_pretrained(save_path)
            if self.tokenizer is not None:
                self.tokenizer.vocab.save_pretrained(save_path)
        elif save_mode == 'pth':
            if not save_path.endswith('pth'):
                save_path += '/' + time.strftime(str(self.module.__class__.__name__) + '_%m%d_%H%M%S.pth')
            torch.save(self.module.state_dict(), save_path)
        else:
            raise ValueError("The save mode does not exist")

    def _train_collate_fn(self, batch):
        return default_collate(batch)

    def _evaluate_collate_fn(self, batch):
        return default_collate(batch)

    def _prepare_train_begin(self, **kwargs):
        pass

    def _on_train_begin(self, **kwargs):
        pass

    def _finish_train_begin(self, **kwargs):
        pass

    def _prepare_train_begin_record(self, **kwargs):
        pass

    def _on_train_begin_record(self, **kwargs):
        pass

    def _finish_train_begin_record(self, **kwargs):
        pass

    def _prepare_epoch_begin(self, **kwargs):
        pass

    def _on_epoch_begin(self, **kwargs):
        pass

    def _finish_epoch_begin(self, **kwargs):
        pass

    def _prepare_epoch_begin_record(self, **kwargs):
        pass

    def _on_epoch_begin_record(self, **kwargs):
        pass

    def _finish_epoch_begin_record(self, **kwargs):
        pass

    def _prepare_step_begin(self, **kwargs):
        pass

    def _on_step_begin(self, **kwargs):
        pass

    def _finish_step_begin(self, **kwargs):
        pass

    def _prepare_step_begin_record(self, **kwargs):
        pass

    def _on_step_begin_record(self, **kwargs):
        pass

    def _finish_step_begin_record(self, **kwargs):
        pass

    def _prepare_compute_loss(self, **kwargs):
        pass

    def _compute_loss(self, **kwargs):
        pass

    def _finish_compute_loss(self, **kwargs):
        pass

    def _prepare_compute_loss_record(self, **kwargs):
        pass

    def _compute_loss_record(self, **kwargs):
        pass

    def _finish_compute_loss_record(self, **kwargs):
        pass

    def _prepare_backward(self, **kwargs):
        pass

    def _on_backward(self,):
        pass

    def _finish_backward(self, **kwargs):
        pass

    def _prepare_backward_record(self, **kwargs):
        pass

    def _on_backward_record(self, **kwargs):
        pass

    def _finish_backward_record(self, **kwargs):
        pass

    def _prepare_optimize(self, **kwargs):
        pass

    def _on_optimize(self, **kwargs):
        pass

    def _finish_optimize(self, **kwargs):
        pass

    def _prepare_optimize_record(self, **kwargs):
        pass

    def _on_optimize_record(self, **kwargs):
        pass

    def _finish_optimize_record(self, **kwargs):
        pass

    def _prepare_step_end(self, **kwargs):
        pass

    def _on_step_end(self, **kwargs):
        pass

    def _finish_step_end(self, **kwargs):
        pass

    def _prepare_step_end_record(self, **kwargs):
        pass

    def _on_step_end_record(self, **kwargs):
        pass

    def _finish_step_end_record(self, **kwargs):
        pass

    def _prepare_epoch_end(self, **kwargs):
        pass

    def _on_epoch_end(self, **kwargs):
        pass

    def _finish_epoch_end(self, **kwargs):
        pass

    def _prepare_epoch_end_record(self, **kwargs):
        pass

    def _on_epoch_end_record(self, **kwargs):
        pass

    def _finish_epoch_end_record(self, **kwargs):
        pass

    def _prepare_train_end(self, **kwargs):
        pass

    def _on_train_end(self, **kwargs):
        pass

    def _finish_train_end(self, **kwargs):
        pass

    def _prepare_train_end_record(self, **kwargs):
        pass

    def _on_train_end_record(self, **kwargs):
        pass

    def _finish_train_end_record(self, **kwargs):
        pass

    def _prepare_fit(self, **kwargs):
        pass

    def fit(self, **kwargs):
        pass

    def _finish_fit(self, **kwargs):
        pass

    def _prepare_evaluate(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass

    def _finish_evaluate(self, **kwargs):
        pass

    def _on_evaluate_epoch_begin_record(self, **kwargs):
        pass

    def _on_evaluate_epoch_end(self, **kwargs):
        pass

    def _on_evaluate_epoch_end_record(self, **kwargs):
        pass

    def _on_evaluate_end(self, **kwargs):
        pass

    def _on_evaluate_end_record(self, **kwargs):
        pass

    def _get_module_inputs_on_train(self):
        pass

    def _get_module_label_on_train(self):
        pass

    def _get_module_inputs_on_eval(self):
        pass

    def _get_module_label_on_eval(self):
        pass
