from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    SparseAdam,
    Adamax,
    ASGD,
    LBFGS,
    RMSprop,
    Rprop,
    SGD
)
from torch.optim import Optimizer
from transformers import AdamW


all_optimizers_dict = dict(
    adadelta=Adadelta,
    adagrad=Adagrad,
    adam=Adam,
    sparseadam=SparseAdam,
    adamax=Adamax,
    asgd=ASGD,
    lbfgs=LBFGS,
    rmsprop=RMSprop,
    rprop=Rprop,
    sgd=SGD,
    adamw=AdamW
)


def get_optimizer(optimizer, module, lr=False, params=None):

    if params is None:
        params_ = (p for p in module.parameters() if p.requires_grad)
    else:
        params_ = params

    if isinstance(optimizer, str):
        optimizer = all_optimizers_dict[optimizer](params_)
    elif type(optimizer).__name__ == 'type' and issubclass(optimizer, Optimizer):
        optimizer = optimizer(params_)
    elif isinstance(optimizer, Optimizer):
        if params is not None:
            optimizer.param_groups = params
    else:
        raise ValueError("The optimizer type does not exist")

    if lr is not False:
        for param_groups_ in optimizer.param_groups:
            param_groups_['lr'] = lr

    return optimizer

def get_default_optimizer(module, module_name='bert', **kwargs):
    module_name = module_name.lower()

    if module_name == 'bert':
        return get_default_bert_optimizer(module, **kwargs)
    elif module_name == 'crf_bert':
        return get_default_crf_bert_optimizer(module, **kwargs)
    else:
        raise ValueError("The default optimizer does not exist")


def get_default_bert_optimizer(
    module,
    lr: float = 3e-5,
    eps: float = 1e-6,
    correct_bias: bool = True,
    weight_decay: float = 1e-3,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay},
        {"params": [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=lr,
                      eps=eps,
                      correct_bias=correct_bias,
                      weight_decay=weight_decay)
    return optimizer


def get_default_crf_bert_optimizer(
    module,
    lr: float = 2e-5,
    crf_lr: float = 2e-3,
    eps: float = 1e-6,
    correct_bias: bool = True,
    weight_decay: float = 1e-2,
):
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(module.bert.named_parameters())
    crf_param_optimizer = list(module.crf.named_parameters())
    linear_param_optimizer = list(module.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': weight_decay, 'lr': lr},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
        'lr': lr},
        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': weight_decay, 'lr': crf_lr},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
        'lr': crf_lr},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': weight_decay, 'lr': crf_lr},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
        'lr': crf_lr}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      eps=eps,
                      correct_bias=correct_bias)

    return optimizer
