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

import torch
import warnings
import numpy as np

from torch.utils.data import DataLoader
from ark_nlp.factory.task.base._task import Task
from torch.utils.data._utils.collate import default_collate

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


class TaskMixin(Task):
    def on_train_begin(self,
                       train_data,
                       epoch_num,
                       batch_size,
                       gradient_accumulation_step,
                       worker_num=0,
                       train_to_device_cols=None,
                       **kwargs):
        # 设置categories
        if hasattr(train_data, 'categories'):
            self.categories = train_data.categories

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

        self.handler.epoch_step_num = len(train_generator) // gradient_accumulation_step

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
        if self.ema:
            self.ema.update(self.module.parameters())

        # 更新学习率
        if self.scheduler:
            self.scheduler.step()

        # 清空梯度
        self.optimizer.zero_grad()

        return self.optimizer

    def on_optimize_record(self, **kwargs):
        self.handler.global_step += 1

        return self.logs

    def on_step_end_record(self, validation_data, step, **kwargs):

        if (step + 1) % self.handler.gradient_accumulation_step == 0:

            # tensorboar记录参数信息
            if self.tb_writer and self.scheduler:
                self.tb_writer.add_scalar("learning_rate",
                                          self.scheduler.get_last_lr()[0],
                                          self.handler.global_step)

            # wandb记录参数信息
            if self.do_wandb_logging and self.scheduler:
                wandb.log({
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "global_step": self.handler.global_step,
                })

            # 记录训练信息
            if self.handler.logging_step > 0 and self.handler.global_step % self.handler.logging_step == 0:

                print('[{}/{}], loss is:{:.6f}'.format(
                    self.handler.epoch_step_num,
                    (step + 1) // self.handler.gradient_accumulation_step,
                    (self.logs['global_loss'] - self.logs['logging_loss']) /
                    self.handler.logging_step))

                # tensorboar记录训练loss信息
                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        "training_loss",
                        (self.logs['global_loss'] - self.logs['logging_loss']) /
                        self.handler.logging_step,
                        self.handler.global_step,
                    )

                # wandb记录训练loss信息
                if self.do_wandb_logging:
                    wandb.log({
                        "training_loss":
                        (self.logs['global_loss'] - self.logs['logging_loss']) /
                        self.handler.logging_step,
                        "global_step":
                        self.handler.global_step,
                    })

                self.logs['logging_loss'] = self.logs['global_loss']

            # 保存模型
            if self.handler.save_step > 0 and self.handler.global_step % self.handler.save_step == 0:
                self.save(self.handler.output_dir,
                          f"checkpoint-step-{self.handler.global_step}", **kwargs)

            # 评估模型
            if self.handler.evaluate_during_training_step > 0 and self.handler.global_step % self.handler.evaluate_during_training_step == 0:

                self.evaluate(validation_data, **kwargs)

                # 保存最佳模型
                if self.handler.do_save_best_module and self.handler.save_best_module_metric is not None:

                    if (self.handler.is_minimize_metric
                            and self.evaluate_logs[self.handler.save_best_module_metric] <
                            self.handler.best_score
                        ) or (not self.handler.is_minimize_metric
                              and self.evaluate_logs[self.handler.save_best_module_metric]
                              > self.handler.best_score):

                        self.handler.best_score = self.evaluate_logs[
                            self.handler.save_best_module_metric]

                        if self.handler.do_early_stopping:
                            self.handler.early_stopping_counter = 0
                        else:
                            self.save(self.handler.output_dir, "checkpoint-best",
                                      **kwargs)
                    else:
                        if self.handler.do_early_stopping:
                            self.handler.early_stopping_counter += 1
                            print('EarlyStopping counter: {} out of {}'.format(
                                self.handler.early_stopping_counter,
                                self.handler.early_stopping_patience))
                            if self.handler.early_stopping_patience <= self.handler.early_stopping_counter:
                                self.handler.should_training_stop = True
                                self.handler.should_epoch_stop = True

                # tensorboar记录评估信息
                if self.tb_writer:
                    for name, metric in self.evaluate_logs.items():
                        if name == 'loss':
                            name = 'evaluating_loss'
                        if type(metric) == float or type(metric) == int or type(
                                metric) == np.float64:
                            self.tb_writer.add_scalar(name, metric,
                                                      self.handler.global_step)

                # wanbd记录评估信息
                if self.do_wandb_logging:
                    for name, metric in self.evaluate_logs.items():
                        if name == 'loss':
                            name = 'evaluating_loss'
                        if type(metric) == float or type(metric) == int or type(
                                metric) == np.float64:
                            wandb.log({name: metric})

            # tensorboar flush
            if self.tb_writer:
                self.tb_writer.flush()

        return self.logs

    def on_epoch_end_record(self, epoch, **kwargs):

        if self.handler.do_evaluate_per_epoch_end:
            print('epoch:[{}], train loss is:{:.6f} \n'.format(
                epoch + 1, self.logs['global_loss'] / self.handler.global_step))

        if self.handler.do_save_per_epoch_end:
            self.save(self.handler.output_dir, f"checkpoint-epoch-{epoch+1}", **kwargs)

        return self.logs

    def on_evaluate_begin(self,
                          validation_data,
                          evaluate_batch_size,
                          shuffle=False,
                          worker_num=0,
                          evaluate_to_device_cols=None,
                          **kwargs):
        # 设置categories
        if not hasattr(self, 'categories') and hasattr(validation_data, 'categories'):
            self.categories = validation_data.categories

        # 设置 self.id2cat 和 self.cat2id
        if not hasattr(self, 'id2cat') and hasattr(validation_data, 'id2cat'):
            self.id2cat = validation_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in validation_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在验证集获取信息
        if self.class_num is None:
            if hasattr(validation_data, 'class_num'):
                self.class_num = validation_data.class_num
            else:
                warnings.warn("The class_num is None.")

        if evaluate_to_device_cols is None:
            self.evaluate_to_device_cols = validation_data.to_device_cols
        else:
            self.evaluate_to_device_cols = evaluate_to_device_cols

        generator = DataLoader(validation_data,
                               batch_size=evaluate_batch_size,
                               shuffle=shuffle,
                               num_workers=worker_num,
                               collate_fn=self._evaluate_collate_fn)

        if self.ema:
            self.ema.store(self.module.parameters())
            self.ema.copy_to(self.module.parameters())

        if self.metric:
            self.metric.reset()

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

    def on_evaluate_epoch_end(self, epoch_step_num, evaluate_verbose=True, **kwargs):

        if 'loss' in self.evaluate_logs:
            self.evaluate_logs['loss'] = self.evaluate_logs['loss'] / epoch_step_num

        if self.metric:
            self.evaluate_logs.update(self.metric.result(categories=self.categories))

        if evaluate_verbose:
            self.log_evaluation()

    def on_evaluate_end(self, **kwargs):

        if self.ema:
            self.ema.restore(self.module.parameters())

        self.module.train()

        return None

    def _train_collate_fn(self, batch):
        return default_collate(batch)

    def _evaluate_collate_fn(self, batch):
        return default_collate(batch)
