# Copyright (c) 2022 DataArk Authors. All Rights Reserved.
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
import time
import torch
import warnings

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


class TaskMixin(object):
    def on_train_begin(self,
                       train_data,
                       epoch_num,
                       batch_size,
                       gradient_accumulation_step,
                       worker_num=0,
                       train_to_device_cols=None,
                       **kwargs):
        # 设置 self.id2cat 和 self.cat2id
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        # 获s获取放置到GPU的变量名称列表
        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(train_data,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=worker_num,
                                     collate_fn=self._train_collate_fn)

        self.epoch_step_num = len(train_generator) // gradient_accumulation_step

        self._set_optimizer(**kwargs)
        self.optimizer.zero_grad()

        self._set_scheduler(epoch_num, **kwargs)

        return train_generator

    def on_epoch_begin(self, **kwargs):
        self.module.train()

        return None

    def get_module_inputs_on_train(self, inputs, **kwargs):
        """模型输入处理方法"""
        for col in self.train_to_device_cols:
            if type(inputs[col]) is torch.Tensor:
                inputs[col] = inputs[col].to(self.device)
            else:
                warnings.warn(f"The {col} is not Tensor.\n")

        return inputs

    def get_module_outputs_on_train(self, inputs, **kwargs):
        return self.module(**inputs)

    def get_train_loss(self, inputs, outputs, **kwargs):
        """获取训练阶段损失的方法"""
        if type(outputs) == tuple:
            if len(outputs) > 2:
                logits, loss, *_ = outputs
            else:
                logits, loss = outputs
        else:
            logits = outputs
            # 计算损失
            loss = self.compute_loss(inputs, logits, **kwargs)

        return logits, loss

    def compute_loss(self, inputs, logits, **kwargs):
        """计算损失的方法"""
        loss = self.loss_function(logits, inputs['label_ids'])

        return loss

    def on_backward(self, loss, gradient_accumulation_step=1, **kwargs):
        # 如果GPU数量大于1
        if self.gpu_num > 1:
            loss = loss.mean()

        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_step > 1:
            loss = loss / gradient_accumulation_step

        loss.backward()

        return loss

    def on_backward_record(self, loss, **kwargs):
        self.logs['global_loss'] += loss.item()

        return self.logs

    def on_optimize(self, grad_clip=None, **kwargs):
        # 梯度裁剪
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), grad_clip)

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

        return self.optimizer

    def on_optimize_record(self, **kwargs):
        self.logs['global_step'] += 1

        return self.logs

    def on_step_end_record(self, validation_data, step, **kwargs):

        # 打印训练信息
        if self.state.logging_step > 0 and self.logs[
                'global_step'] % self.state.logging_step == 0:

            print('[{}/{}], loss is:{:.6f}'.format(
                step, self.epoch_step_num, self.logs['epoch_loss'] -
                self.logs['logging_loss'] / self.state.logging_step))

            self.logs['logging_loss'] = self.logs['global_loss']

        # 保存模型
        if self.state.save_step > 0 and self.logs[
                'global_step'] % self.state.save_step == 0:
            os.makedirs(self.state.output_dir, exist_ok=True)
            self.save(self.state.output_dir)

        # 评估模型
        if self.state.evaluate_during_training_step > 0 and self.logs[
                'global_step'] % self.state.evaluate_during_training_step == 0:
            self.evaluate(validation_data, **kwargs)

            if self.evaluate_logs['evaluate_metric'] > self.state.best_evaluate_metric:
                self.state.best_evaluate_metric = self.evaluate_logs['evaluate_metric']
                os.makedirs(self.state.output_dir, exist_ok=True)
                self.save(self.state.output_dir)

        return self.logs

    def on_epoch_end_record(self, epoch, verbose=True, **kwargs):

        if verbose:
            print('epoch:[{}],train loss is:{:.6f} \n'.format(
                epoch, self.logs['epoch_loss'] / self.logs['epoch_step']))

        self.logs['epoch_loss'] = 0.0
        self.logs['epoch_step'] = 0.0

        return self.logs

    def on_evaluate_begin(self,
                          validation_data,
                          evaluate_batch_size,
                          shuffle=False,
                          worker_num=0,
                          evaluate_to_device_cols=None,
                          **kwargs):
        if evaluate_to_device_cols is None:
            self.evaluate_to_device_cols = validation_data.to_device_cols
        else:
            self.evaluate_to_device_cols = evaluate_to_device_cols

        generator = DataLoader(validation_data,
                               batch_size=evaluate_batch_size,
                               shuffle=shuffle,
                               num_workers=worker_num,
                               collate_fn=self._evaluate_collate_fn)

        if self.ema_decay:
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())

        self.module.eval()

        return generator

    def get_module_inputs_on_evaluate(self, inputs, **kwargs):
        for col in self.evaluate_to_device_cols:
            if type(inputs[col]) is torch.Tensor:
                inputs[col] = inputs[col].to(self.device)
            else:
                warnings.warn(f"The {col} is not Tensor.\n")

        return inputs

    def get_module_outputs_on_evaluate(self, inputs, **kwargs):
        return self.module(**inputs)

    def on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['loss'] += loss.item()

        self.evaluate_logs['example_num'] += len(inputs['label_ids'])
        self.evaluate_logs['step'] += 1

        return None

    def get_evaluate_loss(self, inputs, outputs, **kwargs):
        if type(outputs) == tuple:
            if len(outputs) > 2:
                logits, loss, *_ = outputs
            else:
                logits, loss = outputs
        else:
            logits = outputs
            # 计算损失
            loss = self.compute_loss(inputs, logits, **kwargs)

        return logits, loss

    def on_evaluate_epoch_end(self, evaluate_verbose=True, **kwargs):

        if evaluate_verbose:
            print("********** Evaluating Done **********\n")
            print('loss is:{:.6f}'.format(self.evaluate_logs['loss'] /
                                          self.evaluate_logs['step']))
        return self.evaluate_logs

    def on_evaluate_end(self, evaluate_save=False, save_module_path=None, **kwargs):
        if evaluate_save:
            if save_module_path is None:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')

                prefix = './checkpoint/' + str(self.module.__class__.__name__)
                save_module_path = time.strftime(prefix + '_%m%d_%H%M%S.pth')

            torch.save(self.module.state_dict(), save_module_path)

        if self.ema_decay:
            self.ema.restore(self.module.parameters())

        return None

    def _train_collate_fn(self, batch):
        return default_collate(batch)

    def _evaluate_collate_fn(self, batch):
        return default_collate(batch)
