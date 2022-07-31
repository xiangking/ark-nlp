import torch
import warnings
from torch.utils.data import DataLoader

from ark_nlp.factory.optimizer import get_optimizer


class AWP(object):
    """
    基于AWP算法的攻击机制

    Args:
        module (:obj:`torch.nn.Module`): 模型

    Reference:
        [1] [Adversarial weight perturbation helps robust generalization](https://arxiv.org/abs/2004.05884)
    """
    def __init__(self, module):
        self.module = module
        self.param_backup = {}
        self.param_backup_eps = {}
        self.grad_backup = {}

    def attack(self, epsilon=0.001, alpha=1.0, emb_name='weight', is_first_attack=False):
        if alpha == 0: return
        e = 1e-6
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None and emb_name in name:
                # save
                if is_first_attack:
                    self.param_backup[name] = param.data.clone()
                    grad_eps = epsilon * param.abs().detach()
                    self.param_backup_eps[name] = (
                        self.param_backup[name] - grad_eps,
                        self.param_backup[name] + grad_eps,
                    )
                # attack
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = alpha * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.param_backup_eps[name][0]),
                        self.param_backup_eps[name][1])

    def restore(self):
        for name, param in self.module.named_parameters():
            if name in self.param_backup:
                param.data = self.param_backup[name]
        self.param_backup = {}
        self.param_backup_eps = {}

    def backup_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.module.named_parameters():
            if name in self.grad_backup:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}


class AWPAttackMixin(object):
    def _on_train_begin(self,
                        train_data,
                        validation_data,
                        epochs,
                        batch_size,
                        shuffle,
                        awp_k=3,
                        num_workers=0,
                        train_to_device_cols=None,
                        **kwargs):
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

        train_generator = DataLoader(train_data,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     collate_fn=self._train_collate_fn)
        self.train_generator_lenth = len(train_generator)

        self.set_optimizer(**kwargs)
        self.optimizer.zero_grad()

        self.set_scheduler(epochs, batch_size, **kwargs)

        self.module.train()

        self.awp = AWP(self.module)
        self.awp_k = awp_k

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_backward(self,
                     inputs,
                     outputs,
                     logits,
                     loss,
                     gradient_accumulation_steps=1,
                     **kwargs):

        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        self.awp.backup_grad()
        for t in range(self.awp_k):
            self.awp.attack(is_first_attack=(t == 0))
            if t != self.awp_k - 1:
                self.optimizer.zero_grad()
            else:
                self.awp.restore_grad()
                logits = self.module(**inputs)
                _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
                attck_loss.backward()
        self.awp.restore()

        self._on_backward_record(loss, **kwargs)

        return loss
