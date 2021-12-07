import random
import torch
import numpy as np


def set_seed(seed):
    """
    设置随机种子

    Args:
        seed (:obj:`int`): 随机种子
    """  # noqa: ignore flake8"

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
