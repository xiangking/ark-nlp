from torch.optim import Adadelta
from torch.optim import Adagrad
from torch.optim import Adam
from torch.optim import SparseAdam
from torch.optim import Adamax
from torch.optim import ASGD
from torch.optim import LBFGS
from torch.optim import RMSprop
from torch.optim import Rprop
from torch.optim import SGD
from torch.optim import AdamW


all_optimizers_dict = dict(adadelta=Adadelta,
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


def get_optimizer(optimizer, params):

    if isinstance(optimizer, str):
        optimizer = all_optimizers_dict[optimizer](params)
    else:
        optimizer = optimizer(params)

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
    learning_rate: float = 3e-5,
    epsilon: float = 1e-6,
    weight_decay: float = 1e-3,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in module.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            weight_decay
        },
        {
            "params":
            [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay":
            0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=learning_rate,
                      eps=epsilon,
                      weight_decay=weight_decay)
    return optimizer


def get_default_crf_bert_optimizer(
    module,
    learning_rate: float = 2e-5,
    crf_learning_rate: float = 2e-3,
    epsilon: float = 1e-6,
    weight_decay: float = 1e-2,
):
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(module.bert.named_parameters())
    crf_param_optimizer = list(module.crf.named_parameters())
    linear_param_optimizer = list(module.classifier.named_parameters())
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        weight_decay,
        'lr':
        learning_rate
    }, {
        'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0,
        'lr':
        learning_rate
    }, {
        'params':
        [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        weight_decay,
        'lr':
        crf_learning_rate
    }, {
        'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0,
        'lr':
        crf_learning_rate
    }, {
        'params':
        [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        weight_decay,
        'lr':
        crf_learning_rate
    }, {
        'params':
        [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0,
        'lr':
        crf_learning_rate
    }]
    optimizer = AdamW(optimizer_grouped_parameters, eps=epsilon)

    return optimizer


def get_w2ner_model_optimizer(dl_module,
                              learning_rate: float = 1e-3,
                              bert_learning_rate: float = 5e-6,
                              weight_decay=0.0):
    bert_params = set(dl_module.bert.parameters())
    other_params = list(set(dl_module.parameters()) - bert_params)
    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {
            'params': [
                p for n, p in dl_module.bert.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'lr':
            bert_learning_rate,
            'weight_decay':
            weight_decay
        },
        {
            'params': [
                p for n, p in dl_module.bert.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'lr':
            bert_learning_rate,
            'weight_decay':
            weight_decay
        },
        {
            'params': other_params,
            'lr': learning_rate,
            'weight_decay': weight_decay
        },
    ]

    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    return optimizer
