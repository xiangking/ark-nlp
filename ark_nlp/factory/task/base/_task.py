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

from tqdm import tqdm
from collections import defaultdict
from torch.optim import Optimizer
from torch.utils.data._utils.collate import default_collate
from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.lr_scheduler import get_scheduler
from ark_nlp.factory.utils.ema import EMA


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
        self.ema_decay = ema_decay
        if self.ema_decay:
            self.ema = EMA(self.module.parameters(), decay=self.ema_decay)

        # 设置callbacks
        self.callbacks = [] if callbacks is None else [
            callback() for callback in callbacks
        ]

    def _set_loss_function(self, loss_function):
        if loss_function is None:
            self.loss_function = get_loss(self.default_loss_function)
        elif isinstance(loss_function, str) or isinstance(loss_function, object):
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
            training_step_num = self.epoch_step_num * epoch_num
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

        train_generator = self._on_train_begin(train_data, validation_data, **kwargs)

        for epoch in range(epoch_num):

            self._on_epoch_begin(**kwargs)

            for step, inputs in enumerate(tqdm(train_generator)):

                self._on_step_begin(epoch, step, inputs, **kwargs)

                # input处理和设备转移
                inputs = self._get_module_inputs_on_train(inputs, **kwargs)

                # forward
                outputs = self._get_module_outputs_on_train(inputs, **kwargs)

                # 计算损失
                logits, loss = self._get_train_loss(inputs, outputs, **kwargs)

                # loss backword
                loss = self._on_backward(inputs, outputs, logits, loss, **kwargs)

                if (step + 1) % gradient_accumulation_step == 0:

                    # optimize
                    self._on_optimize(inputs, outputs, logits, loss, **kwargs)

                # setp evaluate
                self._on_step_end(step, inputs, outputs, logits, loss, **kwargs)

            self._on_epoch_end(epoch, **kwargs)

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

    def _on_epoch_begin(self, **kwargs):

        kwargs = self.prepare_epoch_begin(**kwargs)

        self.on_epoch_begin(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_begin') and callable(callback.on_epoch_begin):
                callback.on_epoch_begin(module=self.module,
                                        tokenizer=self.tokenizer,
                                        optimizer=self.optimizer,
                                        scheduler=self.scheduler,
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

    def _get_module_inputs_on_train(self, inputs, **kwargs):
        """模型输入处理阶段"""
        kwargs['inputs'] = inputs
        inputs = self.get_module_inputs_on_train(**kwargs)

        return inputs

    def get_module_inputs_on_train(self, **kwargs):
        return None

    def _get_module_outputs_on_train(self, inputs, **kwargs):

        kwargs['inputs'] = inputs
        outputs = self.get_module_outputs_on_train(**kwargs)

        return outputs

    def get_module_outputs_on_train(self, inputs, **kwargs):
        return self.module(**inputs)

    def _get_train_loss(self, inputs, outputs, **kwargs):
        """获取训练阶段损失阶段"""
        kwargs['inputs'] = inputs
        kwargs['outputs'] = outputs

        logits, loss = self.get_train_loss(**kwargs)

        return logits, loss

    def get_train_loss(self, **kwargs):
        return None, None

    def compute_loss(self, **kwargs):
        return None

    def _on_backward(self, inputs, outputs, logits, loss, **kwargs):
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

    def _on_optimize(self, inputs, outputs, logits, loss, **kwargs):

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

    def _on_step_end(self, step, inputs, outputs, logits, loss, **kwargs):

        kwargs['step'] = step
        kwargs['inputs'] = inputs
        kwargs['outputs'] = outputs
        kwargs['logits'] = logits
        kwargs['loss'] = loss

        kwargs = self.prepare_step_end(**kwargs)

        self.on_step_end(**kwargs)

        for callback in self.callbacks:
            if hasattr(callback, 'on_step_end') and callable(callback.on_step_end):
                callback.on_step_end(module=self.module,
                                     tokenizer=self.tokenizer,
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
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

    def evaluate(self, validation_data, *, evaluate_batch_size=16, **kwargs):
        """
        验证方法
        
        Args:
            validation_data (ark_nlp dataset): 训练的batch文本
            evaluate_batch_size (int, optional): 验证阶段batch大小, 默认值为16
            **kwargs (optional): 其他可选参数
        """  # noqa: ignore flake8"

        self.evaluate_logs = defaultdict(int)
        kwargs['evaluate_batch_size'] = evaluate_batch_size

        evaluate_generator = self._on_evaluate_begin(validation_data, **kwargs)

        with torch.no_grad():

            self._on_evaluate_epoch_begin(**kwargs)

            for step, inputs in enumerate(evaluate_generator):

                inputs = self._get_module_inputs_on_evaluate(inputs, **kwargs)

                # forward
                outputs = self._get_module_outputs_on_evaluate(inputs, **kwargs)

                self._on_evaluate_step_end(step, inputs, outputs, **kwargs)

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

    def _on_evaluate_step_end(self, step, inputs, outputs, **kwargs):

        kwargs['step'] = step
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
        if self.tokenizer is not None:
            self.tokenizer.vocab.save_pretrained(save_path)
        if self.cat2id is not None:
            with open(os.path.join(save_path, 'cat2id.json'), 'w') as f:
                json.dump(self.cat2id, f)

        if save_mode == 'huggingface':
            self.module.save_pretrained(save_path)
        elif save_mode == 'pth':
            if not save_path.endswith('pth'):
                save_path += '/' + time.strftime(
                    str(self.module.__class__.__name__) + '_%m%d_%H%M%S.pth')
            torch.save(self.module.state_dict(), save_path)
        else:
            raise ValueError("The save mode does not exist")

        return save_path
