import torch
import warnings
from torch.utils.data import DataLoader

from ark_nlp.factory.optimizer import get_optimizer


class FreeLB(object):
    """
    基于FreeLB算法的攻击机制
        1. attack the same as PGD
        2. restore is different
        原始论文: 第一次attack，使用随机初始化的扰动。我们使用初始的梯度值

    Args:
        module (:obj:`torch.nn.Module`): 模型

    Reference:
        [1] https://github.com/zhuchen03/FreeLB
        [2] https://www.kaggle.com/code/tsaivincent/at-pure
    """

    def __init__(self, module):
        self.module = module
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(
        self,
        epsilon=1.,
        alpha=0.3,
        emb_name='word_embeddings',
        is_first_attack=False
    ):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self, emb_name='word_embeddings'):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self, emb_name='word_embeddings'):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                # K次扰动的平均梯度值（current_grad） + 原始梯度（backup_grad）
                param.grad = param.grad + self.grad_backup[name]


class FreeLBAttackMixin(object):
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

        self.freelb = FreeLB(self.module)
        self.freelb_k = 3

        self._on_train_begin_record(**kwargs)

        return train_generator

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

        self.freelb.backup_grad()
        for t in range(self.freelb_k):
            self.freelb.attack(is_first_attack=(t == 0))

            if t == 0:   ###原论文是随机初始化，扰动过程中不包含初始的梯度
                self.optimizer.zero_grad()

            logits = self.module(**inputs)
            _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
            attck_loss = attck_loss / self.freelb_k

            attck_loss.backward()

        self.freelb.restore_grad() # add origin actual gradient, + cls_loss
        self.freelb.restore()

        self._on_backward_record(loss, **kwargs)

        return loss