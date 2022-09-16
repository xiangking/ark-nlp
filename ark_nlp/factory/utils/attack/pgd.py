import torch
import warnings
from torch.utils.data import DataLoader


class PGD(object):
    """
    基于PGD算法的攻击机制

    Args:
        module (torch.nn.Module): 模型

    Example:
        .. code-block::

            pgd = PGD(module)
            K = 3
            for batch_input, batch_label in data:
                # 正常训练
                loss = module(batch_input, batch_label)
                loss.backward() # 反向传播, 得到正常的grad
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K-1:
                        optimizer.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv = module(batch_input, batch_label)
                    loss_adv.backward() # 反向传播, 并在正常的grad基础上, 累加对抗训练的梯度
                pgd.restore() # 恢复embedding参数
                # 梯度下降, 更新参数
                optimizer.step()
                optimizer.zero_grad()

    Reference:
        [1]  https://zhuanlan.zhihu.com/p/91269728
    """

    def __init__(self,
                 module,
                 pgd_epsilon=1.,
                 pgd_alpha=0.3,
                 pgd_emb_name='word_embeddings',
                 **kwargs):
        self.module = module

        self.epsilon = pgd_epsilon
        self.alpha = pgd_alpha
        self.emb_name = pgd_emb_name

        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
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

    def on_train_begin(self,
                       train_data,
                       epoch_num,
                       batch_size,
                       gradient_accumulation_step,
                       worker_num=0,
                       train_to_device_cols=None,
                       pgd_k=3,
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

        self.pgd = PGD(self.module)
        self.pgd_k = pgd_k

        return train_generator

    def on_backward(self, inputs, loss, gradient_accumulation_step=1, **kwargs):
        # 如果GPU数量大于1
        if self.gpu_num > 1:
            loss = loss.mean()

        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_step > 1:
            loss = loss / gradient_accumulation_step

        loss.backward()

        self.pgd.backup_grad()
        for t in range(self.pgd_k):
            self.pgd.attack(is_first_attack=(t == 0))
            if t != self.pgd_k - 1:
                self.optimizer.zero_grad()
            else:
                self.pgd.restore_grad()
                outputs = self.module(**inputs)
                _, attck_loss = self.get_train_loss(inputs, outputs)
                attck_loss.backward()

        self.pgd.restore()

        return loss
