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
import warnings

from tqdm import tqdm
from collections import defaultdict
from torch.optim import Optimizer
from torch.utils.data import DataLoader
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
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

    def __init__(
        self,
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
        **kwargs
    ):
        self.module = module
        self.tokenizer = tokenizer

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.set_loss_function(loss_function)

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

    def set_loss_function(
        self,
        loss_function
    ):
        if loss_function is None:
            self.loss_function = get_loss(self.default_loss_function)
        elif isinstance(loss_function, str) or isinstance(loss_function, object):
            self.loss_function = get_loss(loss_function)
        else:
            raise ValueError("The loss function type does not exist")

        return self.loss_function

    def set_optimizer(
        self,
        learning_rate=None,
        epsilon=None,
        weight_decay=None,
        parameters=None
    ):
        # 通过parameters对optimizer内的参数进行修改
        if isinstance(self.optimizer, Optimizer) and not callable(self.optimizer) and parameters is not None:
            for index, param_group in enumerate(self.optimizer.param_groups):
                for key in (set(self.optimizer.param_groups[index].keys()) - set(parameters[index].keys())):
                    parameters[index][key] = self.optimizer.param_groups[index][key]
            self.optimizer.param_groups = parameters

        # 当parameters未定义，且self.optimizer未被创建时, 自动根据module创建parameters
        if parameters is None and not hasattr(self.optimizer, 'param_groups'):
            parameters = [{"params": [p for p in self.module.parameters() if p.requires_grad]}]

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
                param_group['learning_rate'] = learning_rate

        if epsilon is not None:
            for param_group in self.optimizer.param_groups:
                param_group['epsilon'] = epsilon

        if weight_decay is not None:
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = weight_decay

        return self.optimizer

    def set_scheduler(
        self,
        epochs,
        batch_size,
        **kwargs
    ):
        if self.scheduler is not None:
            training_step_num = self.epoch_step_num * epochs
            self.scheduler = get_scheduler(
                self.scheduler,
                self.optimizer,
                training_step_num,
                **kwargs
            )

        return self.scheduler

    def fit(
        self,
        train_data,
        validation_data=None,
        batch_size=32,
        epoch_num=1,
        gradient_accumulation_steps=1,
        **kwargs
    ):
        """
        训练方法
        
        Args:
            train_data (ark_nlp dataset): 训练的batch文本
            validation_data (ark_nlp dataset): 验证的batch文本
            batch_size (int, optional): batch大小, 默认值为: 32
            epoch_num (int, optional): 训练轮数, 默认值为: 1
            gradient_accumulation_steps (int, optional): 梯度累计数, 默认值为: 1
            **kwargs (optional):
                其他可选参数:
                    learning_rate (float or None, optional): 学习率, 默认值为: None
                    epsilon (float or None, optional): 保持数值稳定性的短浮点类型值, 默认值为: None
                    weight_decay (float or None, optional): 权重衰减系数, 默认值为: None
                    parameters (list or None, optional): 指定优化器需要优化的参数, 默认值为: None
        """  # noqa: ignore flake8"

        self.logs = defaultdict(int)

        train_generator = self._on_train_begin(
            train_data,
            validation_data,
            epoch_num,
            batch_size,
            shuffle=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs
        )

        for epoch in range(epoch_num):

            self._on_epoch_begin(**kwargs)

            for step, inputs in enumerate(tqdm(train_generator)):

                self._on_step_begin(epoch, step, inputs, **kwargs)

                # input处理和设备转移
                inputs = self._get_module_inputs_on_train(inputs, **kwargs)

                # forward
                outputs = self.module(**inputs)

                # 计算损失
                logits, loss = self._get_train_loss(inputs, outputs, **kwargs)

                # loss backword
                loss = self._on_backward(inputs, outputs, logits, loss, **kwargs)

                if (step + 1) % gradient_accumulation_steps == 0:

                    # optimize
                    self._on_optimize(inputs, outputs, logits, loss, **kwargs)

                # setp evaluate
                self._on_step_end(step, inputs, outputs, logits, loss, **kwargs)

            self._on_epoch_end(epoch, **kwargs)

            if validation_data is not None:
                self.evaluate(validation_data, **kwargs)

        self._on_train_end(**kwargs)

    def _on_train_begin(
        self,
        train_data,
        validation_data,
        epochs,
        batch_size,
        shuffle,
        gradient_accumulation_steps,
        num_workers=0,
        train_to_device_cols=None,
        **kwargs
    ):
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._train_collate_fn
        )

        self.epoch_step_num = len(train_generator) // gradient_accumulation_steps

        self.set_optimizer(**kwargs)
        self.optimizer.zero_grad()

        self.set_scheduler(epochs, batch_size, **kwargs)

        self.module.train()

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_train_begin_record(self, **kwargs):
        pass

    def _on_epoch_begin(self, **kwargs):

        self.module.train()

        self._on_epoch_begin_record(**kwargs)

    def _on_epoch_begin_record(self, **kwargs):
        pass

    def _on_step_begin(
        self,
        epoch,
        step,
        inputs,
        **kwargs
    ):
        self._on_step_begin_record(**kwargs)

    def _on_step_begin_record(self, **kwargs):
        pass

    def _get_module_inputs_on_train(
        self,
        inputs,
        **kwargs
    ):
        for col in self.train_to_device_cols:
            if type(inputs[col]) is torch.Tensor:
                inputs[col] = inputs[col].to(self.device)
            else:
                warnings.warn(f"The {col} is not Tensor.\n")

        return inputs

    def _get_train_loss(
        self,
        inputs,
        outputs,
        **kwargs
    ):

        if type(outputs) == tuple:
            if len(outputs) > 2:
                logits, loss, *_ = outputs
            else:
                logits, loss = outputs
        else:
            logits = outputs
            # 计算损失
            loss = self._compute_loss(inputs, logits, **kwargs)

        self._compute_loss_record(**kwargs)

        return logits, loss

    def _compute_loss_record(self, **kwargs):
        pass

    def _compute_loss(
        self,
        inputs,
        logits,
        verbose=True,
        **kwargs
    ):
        loss = self.loss_function(logits, inputs['label_ids'])

        return loss

    def _on_backward(
        self,
        inputs,
        outputs,
        logits,
        loss,
        gradient_accumulation_steps=1,
        **kwargs
    ):

        # 如果GPU数量大于1
        if self.gpu_num > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        self._on_backward_record(loss, **kwargs)

        return loss

    def _on_backward_record(self, loss, **kwargs):
        self.logs['global_loss'] += loss.item()
        self.logs['epoch_loss'] += loss.item()

    def _on_optimize(
        self,
        inputs,
        outputs,
        logits,
        loss,
        grad_clip=None,
        **kwargs
    ):

        # 梯度裁剪
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.module.parameters(),
                grad_clip
            )

        # 更新权值
        self.optimizer.step()

        # EMA更新
        if self.ema_decay:
            self.ema.update(self.module.parameters())

        # 更新学习率
        if self.scheduler:
            self.scheduler.step()

        # 清空梯度
        self.optimizer.zero_grad()

        self._on_optimize_record(inputs, outputs, logits, loss, **kwargs)

    def _on_optimize_record(
        self,
        inputs,
        outputs,
        logits,
        loss,
        **kwargs
    ):
        self.logs['global_step'] += 1
        self.logs['epoch_step'] += 1

    def _on_step_end(
        self,
        step,
        inputs,
        outputs,
        loss,
        verbose=True,
        show_metric_step=100,
        **kwargs
    ):

        if verbose and (step + 1) % show_metric_step == 0:
            print('[{}/{}],train loss is:{:.6f}'.format(
                step,
                self.epoch_step_num,
                self.logs['epoch_loss'] / self.logs['epoch_step']))

        self._on_step_end_record(**kwargs)

    def _on_step_end_record(self, **kwargs):
        pass

    def _on_epoch_end(
        self,
        epoch,
        verbose=True,
        **kwargs
    ):

        if verbose:
            print('epoch:[{}],train loss is:{:.6f} \n'.format(
                epoch,
                self.logs['epoch_loss'] / self.logs['epoch_step']))

    def evaluate(
        self,
        validation_data,
        evaluate_batch_size=16,
        **kwargs
    ):
        """
        验证方法
        
        Args:
            validation_data (ark_nlp dataset): 训练的batch文本
            evaluate_batch_size (int, optional): 验证阶段batch大小, 默认值为16
            **kwargs (optional): 其他可选参数
        """  # noqa: ignore flake8"

        self.evaluate_logs = defaultdict(int)

        evaluate_generator = self._on_evaluate_begin(
            validation_data,
            evaluate_batch_size,
            shuffle=False,
            **kwargs
        )

        with torch.no_grad():

            self._on_evaluate_epoch_begin(**kwargs)

            for step, inputs in enumerate(evaluate_generator):

                inputs = self._get_module_inputs_on_eval(inputs, **kwargs)

                # forward
                outputs = self.module(**inputs)

                self._on_evaluate_step_end(inputs, outputs, **kwargs)

            self._on_evaluate_epoch_end(validation_data, **kwargs)

        self._on_evaluate_end(**kwargs)

    def _on_evaluate_begin(
        self,
        validation_data,
        batch_size,
        shuffle,
        worker_num=0,
        evaluate_to_device_cols=None,
        **kwargs
    ):
        if evaluate_to_device_cols is None:
            self.evaluate_to_device_cols = validation_data.to_device_cols
        else:
            self.evaluate_to_device_cols = evaluate_to_device_cols

        evaluate_generator = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=worker_num,
            collate_fn=self._evaluate_collate_fn
        )

        self.module.eval()

        self._on_evaluate_begin_record(**kwargs)

        return evaluate_generator

    def _on_evaluate_begin_record(self, **kwargs):
        pass

    def _on_evaluate_epoch_begin(self, **kwargs):

        if self.ema_decay:
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())

        self._on_evaluate_epoch_begin_record(**kwargs)

    def _on_evaluate_epoch_begin_record(self, **kwargs):
        pass

    def _get_module_inputs_on_eval(
        self,
        inputs,
        **kwargs
    ):
        for col in self.evaluate_to_device_cols:
            if type(inputs[col]) is torch.Tensor:
                inputs[col] = inputs[col].to(self.device)
            else:
                warnings.warn(f"The {col} is not Tensor.\n")

        return inputs

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['eval_loss'] += loss.item()

        self.evaluate_logs['eval_example'] += len(inputs['label_ids'])
        self.evaluate_logs['eval_step'] += 1

    def _get_evaluate_loss(
        self,
        inputs,
        outputs,
        verbose=True,
        **kwargs
    ):

        if type(outputs) == tuple:
            if len(outputs) > 2:
                logits, loss, *_ = outputs
            else:
                logits, loss = outputs
        else:
            logits = outputs
            # 计算损失
            loss = self._compute_loss(inputs, logits, **kwargs)

        return logits, loss

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        epoch=1,
        evaluate_verbose=True,
        **kwargs
    ):
        if evaluate_verbose:
            print('test loss is:{:.6f}'.format(self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step']))

        self._on_evaluate_epoch_end_record(**kwargs)

    def _on_evaluate_epoch_end_record(self, **kwargs):
        pass

    def _on_evaluate_end(
        self,
        evaluate_save=False,
        save_module_path=None,
        **kwargs
    ):

        if evaluate_save:
            if save_module_path is None:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')

                prefix = './checkpoint/' + str(self.module.__class__.__name__)
                save_module_path = time.strftime(prefix + '_%m%d_%H%M%S.pth')

            torch.save(self.module.state_dict(), save_module_path)

        self._on_evaluate_end_record()

        if self.ema_decay:
            self.ema.restore(self.module.parameters())

    def _on_evaluate_end_record(self, **kwargs):
        pass

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
                save_path += '/' + time.strftime(str(self.module.__class__.__name__) + '_%m%d_%H%M%S.pth')
            torch.save(self.module.state_dict(), save_path)
        else:
            raise ValueError("The save mode does not exist")

    def _on_train_end(self, **kwargs):
        pass

    def _on_train_end_record(self, **kwargs):
        pass

    def _train_collate_fn(self, batch):
        return default_collate(batch)

    def _evaluate_collate_fn(self, batch):
        return default_collate(batch)
