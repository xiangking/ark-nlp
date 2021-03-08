from torch.optim import *
from torch.optim import Optimizer



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
    
    params_flag = True
    
    if params == None:
        params_flag = False
        params = (p for p in module.parameters() if p.requires_grad)
    
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        return all_optimizers_dict[optimizer](params, lr)
    
    elif type(optimizer).__name__ == 'type' and issubclass(optimizer, Optimizer):
        return optimizer(params, lr)
    
    elif isinstance(optimizer, Optimizer):
        if lr != None:
            for param_groups_ in optimizer.param_groups:
                param_groups_['lr'] = lr
                
        if params_flag:
            optimizer.param_groups = params
            
        return optimizer
    
    else:
        raise ValueError("The optimizer type does not exist") 