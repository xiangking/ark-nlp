import torch
import warnings
from torch.utils.data import DataLoader


class FreeLB(object):
    """
    基于FreeLB算法的攻击机制
        1. attack the same as PGD
        2. restore is different
        原始论文: 第一次attack，使用随机初始化的扰动。我们使用初始的梯度值

    Args:
        module (torch.nn.Module): 模型

    Reference:
        [1] https://github.com/zhuchen03/FreeLB
        [2] https://www.kaggle.com/code/tsaivincent/at-pure
    """
    def __init__(self, module, freelb_epsilon=1., freelb_alpha=0.3, freelb_emb_name='word_embeddings'):
        self.module = module

        self.epsilon = freelb_epsilon
        self.alpha = freelb_alpha
        self.emb_name = freelb_emb_name

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
                    param.data = self.project(name, param.data, self.epsilon)

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
            if param.requires_grad and self.emb_name in name:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad and self.emb_name in name:
                # K次扰动的平均梯度值（current_grad） + 原始梯度（backup_grad）
                param.grad = param.grad + self.grad_backup[name]


class FreeLBAttackMixin(object):
    def _on_train_begin(self,
                        train_data,
                        validation_data,
                        epoch_num,
                        batch_size,
                        shuffle,
                        gradient_accumulation_step,
                        worker_num=0,
                        train_to_device_cols=None,
                        freelb_k=3,
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

        # 获取获取放置到GPU的变量名称列表
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

        self.set_optimizer(**kwargs)
        self.optimizer.zero_grad()

        self.set_scheduler(epoch_num, batch_size, **kwargs)

        self.freelb = FreeLB(self.module)
        self.freelb_k = freelb_k

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_backward(self,
                     inputs,
                     outputs,
                     logits,
                     loss,
                     gradient_accumulation_step=1,
                     **kwargs):
        # 如果GPU数量大于1
        if self.gpu_num > 1:
            loss = loss.mean()

        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_step > 1:
            loss = loss / gradient_accumulation_step

        loss.backward()

        self.freelb.backup_grad()
        for t in range(self.freelb_k):
            self.freelb.attack(is_first_attack=(t == 0))

            if t == 0:  ###原论文是随机初始化，扰动过程中不包含初始的梯度
                self.optimizer.zero_grad()

            logits = self.module(**inputs)
            _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
            attck_loss = attck_loss / self.freelb_k

            attck_loss.backward()

        self.freelb.restore_grad()  # add origin actual gradient, + cls_loss
        self.freelb.restore()

        self._on_backward_record(loss, **kwargs)

        return loss
