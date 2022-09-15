import torch
import warnings

from torch.utils.data import DataLoader
from ark_nlp.factory.optimizer import get_optimizer


class AWP(object):
    """
    基于AWP算法的攻击机制

    Args:
        module (torch.nn.Module): 模型

    Reference:
        [1] [Adversarial weight perturbation helps robust generalization](https://arxiv.org/abs/2004.05884)
    """

    def __init__(self,
                 module,
                 awp_epsilon=0.001,
                 awp_alpha=1.0,
                 awp_emb_name='weight',
                 **kwargs):
        self.module = module

        self.epsilon = awp_epsilon
        self.alpha = awp_alpha
        self.emb_name = awp_emb_name

        self.param_backup = {}
        self.param_backup_eps = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        if self.alpha == 0:
            return None
        e = 1e-6
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                # save
                if is_first_attack:
                    self.param_backup[name] = param.data.clone()
                    grad_eps = self.epsilon * param.abs().detach()
                    self.param_backup_eps[name] = (
                        self.param_backup[name] - grad_eps,
                        self.param_backup[name] + grad_eps,
                    )
                # attack
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.alpha * param.grad / (norm1 + e) * (norm2 + e)
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

    def on_train_begin(self,
                       train_data,
                       epoch_num,
                       batch_size,
                       gradient_accumulation_step,
                       worker_num=0,
                       train_to_device_cols=None,
                       awp_k=3,
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

        self.awp = AWP(self.module, **kwargs)
        self.awp_k = awp_k

        return train_generator

    def on_backward(self, inputs, loss, gradient_accumulation_step=1, **kwargs):
        # 如果GPU数量大于1
        if self.gpu_num > 1:
            loss = loss.mean()

        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_step > 1:
            loss = loss / gradient_accumulation_step

        loss.backward()

        self.awp.backup_grad()
        for t in range(self.awp_k):
            self.awp.attack(is_first_attack=(t == 0))
            if t != self.awp_k - 1:
                self.optimizer.zero_grad()
            else:
                self.awp.restore_grad()
                outputs = self.module(**inputs)
                _, attck_loss = self.get_train_loss(inputs, outputs)
                attck_loss.backward()

        self.awp.restore()

        return loss
