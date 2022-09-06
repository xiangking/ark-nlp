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

import os
import json
import time
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from torch.optim import Optimizer
from torch.utils.data._utils.collate import default_collate
from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.lr_scheduler import get_scheduler
from ark_nlp.factory.utils.ema import EMA
from ark_nlp.factory.task.base.task_utils import Handler


class Task(object):
    """
    所有Task类的基类, 封装Task类通用的方法和属性

    Args:
        module: 深度学习模型
        optimizer (str or torch.optim.Optimizer or None, optional): 训练模型使用的优化器名或者优化器对象, 默认值为: None
        loss_function (str or object or None, optional): 训练模型使用的损失函数名或损失函数对象, 默认值为: None
        scheduler (torch.optim.lr_scheduler.LambdaLR, optional): scheduler对象, 默认值为: None
        tokenizer (object or None, optional): 分词器, 默认值为: None
        class_num (int or None, optional): 标签数目, 默认值为: None
        gpu_num (int, optional): GPU数目, 默认值为: 1
        device (torch.device, optional): torch.device对象, 当device为None时, 会自动检测是否有GPU
        cuda_device (int, optional): GPU编号, 当device为None时, 根据cuda_device设置device, 默认值为: 0
        ema_decay (int or None, optional): EMA的加权系数, 默认值为: None
        callbacks (list or None, optional): 回调函数列表, 默认值为: None
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

    def __init__(self,
                 module,
                 optimizer=None,
                 loss_function=None,
                 scheduler=None,
                 tokenizer=None,
                 metric=None,
                 class_num=None,
                 gpu_num=1,
                 device=None,
                 cuda_device=0,
                 ema_decay=None,
                 callbacks=None,
                 **kwargs):
        self.module = module
        self.tokenizer = tokenizer

        self.optimizer = optimizer
        self.scheduler = scheduler
        self._set_loss_function(loss_function)

        self._set_metric(metric)

        self.class_num = class_num

        # 设置device
        self.gpu_num = gpu_num
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

        # 多GPU设置
        if self.gpu_num > 1:
            self.module = torch.nn.DataParallel(self.module)

        # 设置EMA
        self.ema = None
        self.ema_decay = ema_decay
        if self.ema_decay:
            self.ema = EMA(self.module.parameters(), decay=self.ema_decay)

        # 设置callbacks
        self.callbacks = [] if callbacks is None else [
            callback() for callback in callbacks
        ]

    def _set_metric(self, metric):
        if callable(metric):
            self.metric = metric()
        elif isinstance(metric, object):
            self.metric = metric
        else:
            self.metric = None

        return self.metric

    def _set_loss_function(self, loss_function):
        if loss_function is None:
            self.loss_function = get_loss(self.default_loss_function)
        elif isinstance(loss_function, str) or isinstance(loss_function, object):
            if callable(loss_function):
                if type(loss_function) == type:
                    self.loss_function = loss_function()
                else:
                    self.loss_function = loss_function
            else:
                self.loss_function = get_loss(loss_function)
        else:
            raise ValueError("The loss function type does not exist")

        return self.loss_function

    def _set_optimizer(self,
                       learning_rate=None,
                       epsilon=None,
                       weight_decay=None,
                       parameters=None,
                       **kwargs):
        # 通过parameters对optimizer内的参数进行修改
        if isinstance(
                self.optimizer,
                Optimizer) and not callable(self.optimizer) and parameters is not None:
            for index, param_group in enumerate(self.optimizer.param_groups):
                for key in (set(self.optimizer.param_groups[index].keys()) -
                            set(parameters[index].keys())):
                    parameters[index][key] = self.optimizer.param_groups[index][key]
            self.optimizer.param_groups = parameters

        # 当parameters未定义，且self.optimizer未被创建时, 自动根据module创建parameters
        if parameters is None and not hasattr(self.optimizer, 'param_groups'):
            parameters = [{
                "params": [p for p in self.module.parameters() if p.requires_grad]
            }]

        # 当optimizer还未被创建时，该部分代码负责创建optimizer
        if self.optimizer is None:
            self.optimizer = get_optimizer(self.default_optimizer, parameters)
        if isinstance(self.optimizer, str) or callable(self.optimizer):
            self.optimizer = get_optimizer(self.optimizer, parameters)
        # 经过上述判断条件后仍然未创建optimizer, 则抛出相关创建异常
        if not isinstance(self.optimizer, Optimizer):
            raise ValueError("The optimizer type does not exist")

        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

        if epsilon is not None:
            for param_group in self.optimizer.param_groups:
                param_group['eps'] = epsilon

        if weight_decay is not None:
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = weight_decay

        return self.optimizer

    def _set_scheduler(self, epoch_num, **kwargs):
        if self.scheduler is not None:
            training_step_num = self.handler.epoch_step_num * epoch_num
            self.scheduler = get_scheduler(self.scheduler, self.optimizer,
                                           training_step_num, **kwargs)

        return self.scheduler

    def fit(self,
            train_data,
            validation_data=None,
            *,
            epoch_num=1,
            batch_size=32,
            gradient_accumulation_step=1,
            **kwargs):
        """
        训练方法
        
        Args:
            train_data (ark_nlp dataset): 训练的batch文本
            validation_data (ark_nlp dataset): 验证的batch文本
            epoch_num (int, optional): 训练轮数, 默认值为: 1
            batch_size (int, optional): batch大小, 默认值为: 32
            gradient_accumulation_step (int, optional): 梯度累计数, 默认值为: 1
            **kwargs (optional):
                其他可选参数:
                    worker_num (int, optional): 并行训练的worker数, Windows下暂不支持设为大于0, 默认值为: 0
                    train_to_device_cols (list or None, optional): 训练数据集中需要转移到指定device的列名, 默认值为: None
                    learning_rate (float or None, optional): 学习率, 默认值为: None
                    epsilon (float or None, optional): 保持数值稳定性的短浮点类型值, 默认值为: None
                    weight_decay (float or None, optional): 权重衰减系数, 默认值为: None
                    parameters (list or None, optional): 指定优化器需要优化的参数, 默认值为: None
        """  # noqa: ignore flake8"

        self.logs = defaultdict(int)
        kwargs['epoch_num'] = epoch_num
        kwargs['batch_size'] = batch_size
        kwargs['gradient_accumulation_step'] = gradient_accumulation_step

        self.handler = Handler()
        self.handler.update_from_dict(kwargs)

        train_generator = self._on_train_begin(train_data, validation_data, **kwargs)

        for epoch in range(epoch_num):

            self._on_epoch_begin(epoch, **kwargs)

            for step, inputs in enumerate(tqdm(train_generator)):

                self._on_step_begin(epoch, step, inputs, **kwargs)

                # input处理和设备转移
                inputs = self._get_module_inputs_on_train(epoch, step, inputs, **kwargs)

                # forward
                outputs = self._get_module_outputs_on_train(epoch, step, inputs, **kwargs)

                # 计算损失
                logits, loss = self._get_train_loss(epoch, step, inputs, outputs,
                                                    **kwargs)

                # loss backward
                loss = self._on_backward(epoch, step, inputs, outputs, logits, loss,
                                         **kwargs)

                if (step + 1) % gradient_accumulation_step == 0:

                    # optimize
                    self._on_optimize(epoch, step, inputs, outputs, logits, loss,
                                      **kwargs)

                # step evaluate
                self._on_step_end(epoch, step, inputs, outputs, logits, loss,
                                  validation_data, **kwargs)

                if self.handler.should_epoch_stop or self.handler.should_training_stop:
                    break

            self._on_epoch_end(epoch, **kwargs)

            if self.handler.should_training_stop:
                break

            if validation_data is not None:
                self.evaluate(validation_data, **kwargs)

        self._on_train_end(**kwargs)

    def _on_train_begin(self, train_data, validation_data, **kwargs):
        kwargs['train_data'] = train_data
        kwargs['validation_data'] = validation_data

        kwargs = self.prepare_train_begin(**kwargs)

        train_generator = self.on_train_begin(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin') and callable(callback.on_train_begin):
                callback.on_train_begin(module=self.module,
                                        tokenizer=self.tokenizer,
                                        optimizer=self.optimizer,
                                        scheduler=self.scheduler,
                                        handler=self.handler,
                                        logs=self.logs,
                                        **kwargs)

        kwargs = self.finish_train_begin(**kwargs)

        self.prepare_train_begin_record(**kwargs)

        self.on_train_begin_record(**kwargs)

        self.finish_train_begin_record(**kwargs)

        return train_generator

    def prepare_train_begin(self, **kwargs):
        return kwargs

    def on_train_begin(self, **kwargs):
        return None

    def finish_train_begin(self, **kwargs):
        return kwargs

    def prepare_train_begin_record(self, **kwargs):
        return self.logs

    def on_train_begin_record(self, **kwargs):
        return self.logs

    def finish_train_begin_record(self, **kwargs):
        return self.logs

    def _on_epoch_begin(self, epoch, **kwargs):

        kwargs['epoch'] = epoch

        kwargs = self.prepare_epoch_begin(**kwargs)

        self.on_epoch_begin(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_begin') and callable(callback.on_epoch_begin):
                callback.on_epoch_begin(module=self.module,
                                        tokenizer=self.tokenizer,
                                        optimizer=self.optimizer,
                                        scheduler=self.scheduler,
                                        handler=self.handler,
                                        logs=self.logs,
                                        **kwargs)

        kwargs = self.finish_epoch_begin(**kwargs)

        self.prepare_epoch_begin_record(**kwargs)

        self.on_epoch_begin_record(**kwargs)

        self.finish_epoch_begin_record(**kwargs)

        return None

    def prepare_epoch_begin(self, **kwargs):
        return kwargs

    def on_epoch_begin(self, **kwargs):
        return None

    def finish_epoch_begin(self, **kwargs):
        return kwargs

    def prepare_epoch_begin_record(self, **kwargs):
        return self.logs

    def on_epoch_begin_record(self, **kwargs):
        return self.logs

    def finish_epoch_begin_record(self, **kwargs):
        return self.logs

    def _on_step_begin(self, epoch, step, inputs, **kwargs):
        kwargs['epoch'] = epoch
        kwargs['step'] = step
        kwargs['inputs'] = inputs

        kwargs = self.prepare_step_begin(**kwargs)

        self.on_step_begin(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_step_begin') and callable(callback.on_step_begin):
                callback.on_step_begin(module=self.module,
                                       tokenizer=self.tokenizer,
                                       optimizer=self.optimizer,
                                       scheduler=self.scheduler,
                                       handler=self.handler,
                                       logs=self.logs,
                                       **kwargs)

        kwargs = self.finish_step_begin(**kwargs)

        self.prepare_step_begin_record(**kwargs)

        self.on_step_begin_record(**kwargs)

        self.finish_step_begin_record(**kwargs)

        return None

    def prepare_step_begin(self, **kwargs):
        return kwargs

    def on_step_begin(self, **kwargs):
        return None

    def finish_step_begin(self, **kwargs):
        return kwargs

    def prepare_step_begin_record(self, **kwargs):
        return self.logs

    def on_step_begin_record(self, **kwargs):
        return self.logs

    def finish_step_begin_record(self, **kwargs):
        return self.logs

    def _get_module_inputs_on_train(self, epoch, step, inputs, **kwargs):
        """模型输入处理阶段"""

        kwargs['epoch'] = epoch
        kwargs['step'] = step
        kwargs['inputs'] = inputs
        inputs = self.get_module_inputs_on_train(**kwargs)

        return inputs

    def get_module_inputs_on_train(self, **kwargs):
        return None

    def _get_module_outputs_on_train(self, epoch, step, inputs, **kwargs):

        kwargs['epoch'] = epoch
        kwargs['step'] = step
        kwargs['inputs'] = inputs
        outputs = self.get_module_outputs_on_train(**kwargs)

        return outputs

    def get_module_outputs_on_train(self, inputs, **kwargs):
        return self.module(**inputs)

    def _get_train_loss(self, epoch, step, inputs, outputs, **kwargs):
        """获取训练阶段损失阶段"""

        kwargs['epoch'] = epoch
        kwargs['step'] = step
        kwargs['inputs'] = inputs
        kwargs['outputs'] = outputs

        logits, loss = self.get_train_loss(**kwargs)

        return logits, loss

    def get_train_loss(self, **kwargs):
        return None, None

    def compute_loss(self, **kwargs):
        return None

    def _on_backward(self, epoch, step, inputs, outputs, logits, loss, **kwargs):

        kwargs['epoch'] = epoch
        kwargs['step'] = step
        kwargs['inputs'] = inputs
        kwargs['outputs'] = outputs
        kwargs['logits'] = logits
        kwargs['loss'] = loss

        kwargs = self.prepare_backward(**kwargs)

        self.on_backward(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_backward') and callable(callback.on_backward):
                callback.on_backward(module=self.module,
                                     tokenizer=self.tokenizer,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
                                     handler=self.handler,
                                     logs=self.logs,
                                     **kwargs)

        kwargs = self.finish_backward(**kwargs)

        self.prepare_backward_record(**kwargs)

        self.on_backward_record(**kwargs)

        self.finish_backward_record(**kwargs)

        return loss

    def prepare_backward(self, **kwargs):
        return kwargs

    def on_backward(self, **kwargs):
        return None

    def finish_backward(self, **kwargs):
        return kwargs

    def prepare_backward_record(self, **kwargs):
        return self.logs

    def on_backward_record(self, loss, **kwargs):
        return self.logs

    def finish_backward_record(self, **kwargs):
        return self.logs

    def _on_optimize(self, epoch, step, inputs, outputs, logits, loss, **kwargs):

        kwargs['epoch'] = epoch
        kwargs['step'] = step
        kwargs['inputs'] = inputs
        kwargs['outputs'] = outputs
        kwargs['logits'] = logits
        kwargs['loss'] = loss

        kwargs = self.prepare_optimize(**kwargs)

        self.on_optimize(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_optimize') and callable(callback.on_optimize):
                callback.on_optimize(module=self.module,
                                     tokenizer=self.tokenizer,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
                                     handler=self.handler,
                                     logs=self.logs,
                                     **kwargs)

        kwargs = self.finish_optimize(**kwargs)

        self.prepare_optimize_record(**kwargs)

        self.on_optimize_record(**kwargs)

        self.finish_optimize_record(**kwargs)

        return self.optimizer

    def prepare_optimize(self, **kwargs):
        return kwargs

    def on_optimize(self, **kwargs):
        return None

    def finish_optimize(self, **kwargs):
        return kwargs

    def prepare_optimize_record(self, **kwargs):
        return self.logs

    def on_optimize_record(self, **kwargs):
        return self.logs

    def finish_optimize_record(self, **kwargs):
        return self.logs

    def _on_step_end(self, epoch, step, inputs, outputs, logits, loss, validation_data,
                     **kwargs):

        kwargs['epoch'] = epoch
        kwargs['step'] = step
        kwargs['inputs'] = inputs
        kwargs['outputs'] = outputs
        kwargs['logits'] = logits
        kwargs['loss'] = loss
        kwargs['validation_data'] = validation_data

        kwargs = self.prepare_step_end(**kwargs)

        self.on_step_end(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_step_end') and callable(callback.on_step_end):
                callback.on_step_end(module=self.module,
                                     tokenizer=self.tokenizer,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
                                     handler=self.handler,
                                     logs=self.logs,
                                     **kwargs)

        kwargs = self.finish_step_end(**kwargs)

        self.prepare_step_end_record(**kwargs)

        self.on_step_end_record(**kwargs)

        self.finish_step_end_record(**kwargs)

        return None

    def prepare_step_end(self, **kwargs):
        return kwargs

    def on_step_end(self, **kwargs):
        return None

    def finish_step_end(self, **kwargs):
        return kwargs

    def prepare_step_end_record(self, **kwargs):
        return self.logs

    def on_step_end_record(self, **kwargs):
        return self.logs

    def finish_step_end_record(self, **kwargs):
        return self.logs

    def _on_epoch_end(self, epoch, **kwargs):

        kwargs['epoch'] = epoch

        kwargs = self.prepare_epoch_end(**kwargs)

        self.on_epoch_end(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end') and callable(callback.on_epoch_end):
                callback.on_epoch_end(module=self.module,
                                      tokenizer=self.tokenizer,
                                      optimizer=self.optimizer,
                                      scheduler=self.scheduler,
                                      handler=self.handler,
                                      logs=self.logs,
                                      **kwargs)

        kwargs = self.finish_epoch_end(**kwargs)

        self.prepare_epoch_end_record(**kwargs)

        self.on_epoch_end_record(**kwargs)

        self.finish_epoch_end_record(**kwargs)

        return None

    def prepare_epoch_end(self, **kwargs):
        return kwargs

    def on_epoch_end(self, **kwargs):
        return None

    def finish_epoch_end(self, **kwargs):
        return kwargs

    def prepare_epoch_end_record(self, **kwargs):
        return self.logs

    def on_epoch_end_record(self, **kwargs):
        return self.logs

    def finish_epoch_end_record(self, **kwargs):
        return self.logs

    def _on_train_end(self, **kwargs):

        kwargs = self.prepare_train_end(**kwargs)

        self.on_train_end(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end') and callable(callback.on_train_end):
                callback.on_train_end(module=self.module,
                                      tokenizer=self.tokenizer,
                                      optimizer=self.optimizer,
                                      scheduler=self.scheduler,
                                      handler=self.handler,
                                      logs=self.logs,
                                      **kwargs)

        kwargs = self.finish_train_end(**kwargs)

        self.prepare_train_end_record(**kwargs)

        self.on_train_end_record(**kwargs)

        self.finish_train_end_record(**kwargs)

        return None

    def prepare_train_end(self, **kwargs):
        return kwargs

    def on_train_end(self, **kwargs):
        return None

    def finish_train_end(self, **kwargs):
        return kwargs

    def prepare_train_end_record(self, **kwargs):
        return self.logs

    def on_train_end_record(self, **kwargs):
        return self.logs

    def finish_train_end_record(self, **kwargs):
        return self.logs

    # @torch.no_grad()
    def evaluate(self, validation_data, *, evaluate_batch_size=16, **kwargs):
        """
        验证方法
        
        Args:
            validation_data (ark_nlp dataset): 训练的batch文本
            evaluate_batch_size (int, optional): 验证阶段batch大小, 默认值为16
            **kwargs (optional): 其他可选参数
        """  # noqa: ignore flake8"

        self.evaluate_logs = defaultdict(int)

        kwargs = self.remove_invalid_arguments(kwargs)
        kwargs['evaluate_batch_size'] = evaluate_batch_size

        evaluate_generator = self._on_evaluate_begin(validation_data, **kwargs)

        kwargs['epoch_step_num'] = len(evaluate_generator)

        with torch.no_grad():

            self._on_evaluate_epoch_begin(**kwargs)

            for step, inputs in enumerate(evaluate_generator):

                inputs = self._get_module_inputs_on_evaluate(inputs, **kwargs)

                # forward
                outputs = self._get_module_outputs_on_evaluate(inputs, **kwargs)

                self._on_evaluate_step_end(inputs, outputs, **kwargs)

            self._on_evaluate_epoch_end(validation_data, **kwargs)

        self._on_evaluate_end(**kwargs)

    def _on_evaluate_begin(self, validation_data, **kwargs):

        kwargs['validation_data'] = validation_data

        kwargs = self.prepare_evaluate_begin(**kwargs)

        generator = self.on_evaluate_begin(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_evaluate_begin') and callable(
                    callback.on_evaluate_begin):
                callback.on_evaluate_begin(module=self.module,
                                           tokenizer=self.tokenizer,
                                           optimizer=self.optimizer,
                                           scheduler=self.scheduler,
                                           logs=self.evaluate_logs,
                                           **kwargs)

        kwargs = self.finish_evaluate_begin(**kwargs)

        return generator

    def prepare_evaluate_begin(self, **kwargs):
        return kwargs

    def on_evaluate_begin(self, **kwargs):
        return None

    def finish_evaluate_begin(self, **kwargs):
        return kwargs

    def _on_evaluate_epoch_begin(self, **kwargs):

        kwargs = self.prepare_evaluate_epoch_begin(**kwargs)

        self.on_evaluate_epoch_begin(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_evaluate_epoch_begin') and callable(
                    callback.on_evaluate_epoch_begin):
                callback.on_evaluate_epoch_begin(module=self.module,
                                                 tokenizer=self.tokenizer,
                                                 optimizer=self.optimizer,
                                                 scheduler=self.scheduler,
                                                 logs=self.evaluate_logs,
                                                 **kwargs)

        kwargs = self.finish_evaluate_epoch_begin(**kwargs)

        return None

    def prepare_evaluate_epoch_begin(self, **kwargs):
        return kwargs

    def on_evaluate_epoch_begin(self, **kwargs):
        return None

    def finish_evaluate_epoch_begin(self, **kwargs):
        return kwargs

    def _get_module_inputs_on_evaluate(self, inputs, **kwargs):

        kwargs['inputs'] = inputs
        inputs = self.get_module_inputs_on_evaluate(**kwargs)

        return inputs

    def get_module_inputs_on_evaluate(self, **kwargs):
        return None

    def _get_module_outputs_on_evaluate(self, inputs, **kwargs):

        kwargs['inputs'] = inputs
        outputs = self.get_module_outputs_on_evaluate(**kwargs)

        return outputs

    def get_module_outputs_on_evaluate(self, **kwargs):
        return None

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        kwargs['inputs'] = inputs
        kwargs['outputs'] = outputs

        kwargs = self.prepare_evaluate_step_end(**kwargs)

        self.on_evaluate_step_end(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_evaluate_step_end') and callable(
                    callback.on_evaluate_step_end):
                callback.on_evaluate_step_end(module=self.module,
                                              tokenizer=self.tokenizer,
                                              optimizer=self.optimizer,
                                              scheduler=self.scheduler,
                                              logs=self.evaluate_logs,
                                              **kwargs)

        kwargs = self.finish_evaluate_step_end(**kwargs)

        return None

    def prepare_evaluate_step_end(self, **kwargs):
        return kwargs

    def finish_evaluate_step_end(self, **kwargs):
        return kwargs

    def on_evaluate_step_end(self, **kwargs):
        return None

    def _get_evaluate_loss(self, inputs, outputs, **kwargs):

        kwargs['inputs'] = inputs
        kwargs['outputs'] = outputs

        logits, loss = self.get_evaluate_loss(**kwargs)

        return logits, loss

    def get_evaluate_loss(self, **kwargs):
        return None, None

    def _on_evaluate_epoch_end(self, validation_data, **kwargs):

        kwargs['validation_data'] = validation_data

        kwargs = self.prepare_evaluate_epoch_end(**kwargs)

        self.on_evaluate_epoch_end(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_evaluate_epoch_end') and callable(
                    callback.on_evaluate_epoch_end):
                callback.on_evaluate_epoch_end(module=self.module,
                                               tokenizer=self.tokenizer,
                                               optimizer=self.optimizer,
                                               scheduler=self.scheduler,
                                               logs=self.evaluate_logs,
                                               **kwargs)

        kwargs = self.finish_evaluate_epoch_end(**kwargs)

        return None

    def prepare_evaluate_epoch_end(self, **kwargs):
        return kwargs

    def finish_evaluate_epoch_end(self, **kwargs):
        return kwargs

    def on_evaluate_epoch_end(self, **kwargs):
        return self.evaluate_logs

    def _on_evaluate_end(self, **kwargs):

        kwargs = self.prepare_evaluate_end(**kwargs)

        self.on_evaluate_end(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_evaluate_end') and callable(
                    callback.on_evaluate_end):
                callback.on_evaluate_end(module=self.module,
                                         tokenizer=self.tokenizer,
                                         optimizer=self.optimizer,
                                         scheduler=self.scheduler,
                                         logs=self.evaluate_logs,
                                         **kwargs)

        kwargs = self.finish_evaluate_end(**kwargs)

        return None

    def prepare_evaluate_end(self, **kwargs):
        return kwargs

    def finish_evaluate_end(self, **kwargs):
        return kwargs

    def on_evaluate_end(self, **kwargs):
        return None

    def _train_collate_fn(self, batch):
        return default_collate(batch)

    def _evaluate_collate_fn(self, batch):
        return default_collate(batch)

    def save(self,
             output_dir,
             module_name=None,
             save_mode=None,
             save_format=None,
             **kwargs):
        """
        提供多种方式保存模型
        
        Args:
            output_dir (str): 保存路径
            module_name (str, optional): 模型名称
            save_mode (str, optional):
                保存的方式
                "pretrained"表示会以transformers库保存预训练模型的格式进行保存
                "torch"表示module会以torch.save的方式保存模型权重
            save_format (str, optional): 保存格式, 默认值为: "pth"
        """  # noqa: ignore flake8"

        os.makedirs(output_dir, exist_ok=True)

        if self.ema:
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())

        if save_mode is None:
            save_mode = 'torch'

        if save_format is None:
            save_format = 'pth'

        if save_mode == 'pretrained':
            if module_name:
                output_dir = os.path.join(output_dir, module_name)

            if self.tokenizer is not None:
                self.tokenizer.vocab.save_pretrained(output_dir)

            if self.cat2id is not None:
                with open(os.path.join(output_dir, 'cat2id.json'), 'w') as f:
                    json.dump(self.cat2id, f)

            self.module.save_pretrained(output_dir)

        elif save_mode == 'torch':
            if self.tokenizer is not None:
                self.tokenizer.vocab.save_pretrained(output_dir)

            if self.cat2id is not None:
                with open(os.path.join(output_dir, 'cat2id.json'), 'w') as f:
                    json.dump(self.cat2id, f)

            if module_name is None:
                module_name = time.strftime(
                    str(self.module.__class__.__name__) +
                    '_%m%d_%H%M%S') + '.' + save_format
            else:
                module_name += '.' + save_format

            output_dir = os.path.join(output_dir, module_name)

            torch.save(self.module.state_dict(), output_dir)

        else:
            raise ValueError("The save omde does not exist")

        if self.ema:
            self.ema.restore(self.module.parameters())

    def log_evaluation(self):

        print("\n******************** Evaluating Done ********************\n")

        for name, metric in self.evaluate_logs.items():
            if type(metric) == float or type(metric) == int or type(metric) == np.float64:
                print('{} is: {:.6f}'.format(name, metric))
            else:
                print('{} is: \n{}'.format(name, metric))

    @property
    def metric_names(self):
        if self.metric:
            return self.metric.name
        return None

    def remove_invalid_arguments(self, kwargs):
        for arg_name in ['inputs', 'outputs', 'logits', 'loss']:
            if arg_name in kwargs:
                del kwargs[arg_name]
        return kwargs
