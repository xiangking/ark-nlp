from torch.optim import *
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
    adamw=AdamW)


def get_optimizer(optimizer, module, lr=None, params=None):
    
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        if params is None:
            params = (p for p in module.parameters() if p.requires_grad)
        return all_optimizers_dict[optimizer](params, lr if lr is not None else 1e-3)
    
    elif type(optimizer).__name__ == 'type' and issubclass(optimizer, Optimizer):
        if params is None:
            params = (p for p in module.parameters() if p.requires_grad)
        return optimizer(params, lr if lr is not None else 1e-3)
    
    elif isinstance(optimizer, Optimizer):
        if lr is not None:
            for param_groups_ in optimizer.param_groups:
                param_groups_['lr'] = lr
                
        if params is not None:
            optimizer.param_groups = params
            
        return optimizer
    
    else:
        raise ValueError("The optimizer type does not exist") 


def get_default_optimizer(module, module_name='bert', **kwargs):
    module_name = module_name.lower()

    if module_name == 'bert':
        return get_default_bert_optimizer(module, **kwargs)
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
    