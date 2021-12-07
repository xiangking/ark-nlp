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

import time
import torch
import warnings

from tqdm import tqdm
from torch.utils.data import DataLoader
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.task.base._task import Task


class SequenceClassificationTask(Task):
    """
    序列分类任务的基类
    
    Args:
        module: 深度学习模型
        optimizer: 训练模型使用的优化器名或者优化器对象
        loss_function: 训练模型使用的损失函数名或损失函数对象
        class_num (:obj:`int` or :obj:`None`, optional, defaults to None): 标签数目
        scheduler (:obj:`class`, optional, defaults to None): scheduler对象
        n_gpu (:obj:`int`, optional, defaults to 1): GPU数目
        device (:obj:`class`, optional, defaults to None): torch.device对象，当device为None时，会自动检测是否有GPU
        cuda_device (:obj:`int`, optional, defaults to 0): GPU编号，当device为None时，根据cuda_device设置device
        ema_decay (:obj:`int` or :obj:`None`, optional, defaults to None): EMA的加权系数
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

    def __init__(self, *args, **kwargs):
        super(SequenceClassificationTask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'SequenceLevel'

    def fit(
        self,
        train_data,
        validation_data=None,
        lr=False,
        params=None,
        batch_size=32,
        epochs=1,
        gradient_accumulation_steps=1,
        **kwargs
    ):
        """
        训练方法
        
        Args:
            train_data (:obj:`ark_nlp dataset`): 训练的batch文本
            validation_data (:obj:`ark_nlp dataset`): 验证的batch文本
            lr (:obj:`float` or :obj:`bool`, optional, defaults to False): 学习率
            params (:obj:`str` or :obj:`torch.optim.Optimizer` or :obj:`list` or :obj:`None`, optional, defaults to None): 优化器，可能是名称、对象、参数列表
            batch_size (:obj:`int`, optional, defaults to 32): batch大小
            epochs (:obj:`int`, optional, defaults to 1): 训练轮数
            gradient_accumulation_steps (:obj:`int`, optional, defaults to 1): 梯度累计数
            **kwargs (optional): 其他可选参数
        """  # noqa: ignore flake8"

        self.logs = dict()

        train_generator = self._on_train_begin(
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle=True,
            **kwargs
        )

        for epoch in range(epochs):

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
        batch_size,
        lr,
        params,
        shuffle,
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
        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()

        self.module.train()

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_train_begin_record(self, **kwargs):

        self.logs['global_step'] = 0
        self.logs['global_loss'] = 0

    def _on_epoch_begin(self, **kwargs):

        self.module.train()

        self._on_epoch_begin_record(**kwargs)

    def _on_epoch_begin_record(self, **kwargs):

        self.logs['epoch_loss'] = 0
        # 占位作用，子类仅使用单个指标进行评价，则直接使用该字段即可
        self.logs['epoch_evaluation'] = 0
        self.logs['epoch_step'] = 0

    def _on_step_begin(
        self,
        epoch,
        step,
        inputs,
        **kwargs
    ):
        self._on_step_begin_record(**kwargs)

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
        if self.n_gpu > 1:
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
        show_step=100,
        **kwargs
    ):

        if verbose and (step + 1) % show_step == 0:
            print('[{}/{}],train loss is:{:.6f}'.format(
                step,
                self.train_generator_lenth,
                self.logs['epoch_loss'] / self.logs['epoch_step']))

        self._on_step_end_record(**kwargs)

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
            validation_data (:obj:`ark_nlp dataset`): 训练的batch文本
            evaluate_batch_size (:obj:`int`, optional, defaults to 32): 验证阶段batch大小
            **kwargs (optional): 其他可选参数
        """  # noqa: ignore flake8"

        self.evaluate_logs = dict()

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
        num_workers=0,
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
            num_workers=num_workers,
            collate_fn=self._evaluate_collate_fn
        )

        self.module.eval()

        self._on_evaluate_begin_record(**kwargs)

        return evaluate_generator

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_example'] = 0

    def _on_evaluate_epoch_begin(self, **kwargs):

        if self.ema_decay:
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())

        self._on_evaluate_epoch_begin_record(**kwargs)

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
        is_evaluate_print=True,
        **kwargs
    ):
        if is_evaluate_print:
            print('test loss is:{:.6f}'.format(self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step']))

    def _on_evaluate_end(
        self,
        evaluate_save=False,
        save_module_path=None,
        **kwargs
    ):

        if evaluate_save:
            if save_module_path is None:
                prefix = './checkpoint/' + str(self.module.__class__.__name__) + '_'
                save_module_path = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

            torch.save(self.module.state_dict(), save_module_path)

        self._on_evaluate_end_record()

        if self.ema_decay:
            self.ema.restore(self.module.parameters())
