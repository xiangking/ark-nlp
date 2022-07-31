import torch
import warnings
from torch.utils.data import DataLoader


class PGD(object):
    """
    基于PGD算法的攻击机制

    Args:
        module (:obj:`torch.nn.Module`): 模型

    Examples::

        >>> pgd = PGD(module)
        >>> K = 3
        >>> for batch_input, batch_label in data:
        >>>     # 正常训练
        >>>     loss = module(batch_input, batch_label)
        >>>     loss.backward() # 反向传播，得到正常的grad
        >>>     pgd.backup_grad()
        >>>     # 对抗训练
        >>>     for t in range(K):
        >>>         pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        >>>         if t != K-1:
        >>>             optimizer.zero_grad()
        >>>         else:
        >>>             pgd.restore_grad()
        >>>         loss_adv = module(batch_input, batch_label)
        >>>         loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        >>>     pgd.restore() # 恢复embedding参数
        >>>     # 梯度下降，更新参数
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()

    Reference:
        [1]  https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, module):
        self.module = module
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self,
               epsilon=1.,
               alpha=0.3,
               emb_name='word_embeddings',
               is_first_attack=False):
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

    def backup_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class PGDAttackMixin(object):
    def _on_train_begin(self,
                        train_data,
                        validation_data,
                        epochs,
                        batch_size,
                        shuffle,
                        pgd_k=3,
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

        self.pgd = PGD(self.module)
        self.pgd_k = pgd_k

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

        self.pgd.backup_grad()
        for t in range(self.pgd_k):
            self.pgd.attack(is_first_attack=(t == 0))
            if t != self.pgd_k - 1:
                self.optimizer.zero_grad()
            else:
                self.pgd.restore_grad()
                logits = self.module(**inputs)
                _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
                attck_loss.backward()
        self.pgd.restore()

        self._on_backward_record(loss, **kwargs)

        return loss
