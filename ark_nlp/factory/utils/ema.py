import torch


class EMA(object):
    """
    Maintains (exponential) moving average of a set of parameters.
    使用ema累积模型参数

    Args:
        parameters (:obj:`list`): 需要训练的模型参数
        decay (:obj:`float`): 指数衰减率
        use_num_updates (:obj:`bool`, optional, defaults to True): Whether to use number of updates when computing averages

    Examples::

        >>> ema = EMA(module.parameters(), decay=0.995)
        >>> # Train for a few epochs
        >>> for _ in range(epochs):
        >>>     # 训练过程中，更新完参数后，同步update shadow weights
        >>>     optimizer.step()
        >>>     ema.update(module.parameters())
        >>> # eval前，进行ema的权重替换；eval之后，恢复原来模型的参数
        >>> ema.store(module.parameters())
        >>> ema.copy_to(module.parameters())
        >>> # evaluate
        >>> ema.restore(module.parameters())

    Reference:
        [1]  https://github.com/fadel/pytorch_ema
    """  # noqa: ignore flake8"

    def __init__(
        self,
        parameters,
        decay,
        use_num_updates=True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone()
                                 for param in parameters
                                 if param.requires_grad]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)
