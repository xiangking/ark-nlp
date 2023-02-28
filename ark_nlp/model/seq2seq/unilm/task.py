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
# Author: Chenjie Shen, jimme.shen123@gmail.com
# Status: Active

import torch

from ark_nlp.factory.task.base._task import Task
from ark_nlp.factory.task.base._task_mixin import TaskMixin
from ark_nlp.factory.metric.seq2seq_metric import Seq2SeqMetric
from ark_nlp.model.seq2seq.unilm.predictor import UniLMPredictor
from torch.utils.data._utils.collate import default_collate


class UniLMTask(TaskMixin, Task):
    """
    序列分类任务的基类

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

    def __init__(self, *args, **kwargs):
        super(UniLMTask, self).__init__(*args, **kwargs)
        if hasattr(self.module, 'task') is False:
            self.module.task = 'SequenceLevel'

        if 'metric' not in kwargs:
            self.metric = Seq2SeqMetric()

        # 解码的长度
        self.maxlen = kwargs.get('maxlen', self.tokenizer.max_seq_len // 2)

        #
        self.KG = None

        self.unilm_predictor_instance = UniLMPredictor(module=self.module, tokenizer=self.tokenizer, KG=self.KG,
                             start_id=None, end_id=self.tokenizer.vocab.sep_token_id,
                             maxlen=self.maxlen, device=self.device)

    @property
    def default_optimizer(self):
        return 'adamw'

    @property
    def default_loss_function(self):
        return 'seq2seqce'

    def _train_collate_fn(self, batch):
        input_ids = default_collate([f['input_ids'] for f in batch])
        token_type_ids = default_collate([f['token_type_ids'] for f in batch])
        text = [f['text'] for f in batch]
        label = [f['label'] for f in batch]

        tensors = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'text': text,
            'label': label,
        }
        return tensors

    def _evaluate_collate_fn(self, batch):
        return self._train_collate_fn(batch)

    def compute_loss(self, inputs, outputs, **kwargs):
        loss = self.loss_function(outputs, inputs)
        return loss

    def on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)

            self.evaluate_logs['loss'] += loss.item()

            preds = []
            for text in inputs['text']:
                pred = self.unilm_predictor_instance.predict_one_sample(text, topk=3, min_ends=1)
                preds.append(pred)
            print(preds)
            self.metric.update(preds, inputs['label'])

        return None