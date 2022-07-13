import torch
import warnings
from torch.utils.data import DataLoader

from ark_nlp.factory.optimizer import get_optimizer


class FGM(object):
    """
    基于FGM算法的攻击机制

    Args:
        module (:obj:`torch.nn.Module`): 模型

    Examples::

        >>> # 初始化
        >>> fgm = FGM(module)
        >>> for batch_input, batch_label in data:
        >>>     # 正常训练
        >>>     loss = module(batch_input, batch_label)
        >>>     loss.backward() # 反向传播，得到正常的grad
        >>>     # 对抗训练
        >>>     fgm.attack() # 在embedding上添加对抗扰动
        >>>     loss_adv = module(batch_input, batch_label)
        >>>     loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        >>>     fgm.restore() # 恢复embedding参数
        >>>     # 梯度下降，更新参数
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()

    Reference:
        [1]  https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, module):
        self.module = module
        self.backup = {}

    def attack(
        self,
        epsilon=1.,
        emb_name='word_embeddings'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(
        self,
        emb_name='word_embeddings'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGMAttackMixin(object):

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

        self.fgm = FGM(self.module)

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

        self.fgm.attack()
        logits = self.module(**inputs)
        _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
        attck_loss.backward()
        self.fgm.restore()

        self._on_backward_record(loss, **kwargs)

        return loss
