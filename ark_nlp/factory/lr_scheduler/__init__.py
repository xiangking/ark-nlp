from transformers.optimization import get_constant_schedule
from transformers.optimization import get_constant_schedule_with_warmup
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers.optimization import get_polynomial_decay_schedule_with_warmup


def get_scheduler(scheduler,
                  optimizer,
                  training_step_num,
                  warmup_step: int = 0,
                  warmup_ratio: float = 0.06,
                  cosine_schedule_num_cycles: float = 0.5,
                  polynomial_decay_schedule_lr_end: float = 1e-7,
                  polynomial_decay_schedule_power: float = 1.0):
    """
    加载LR调度器
    Args:
        scheduler (string or torch module): LR调度器名或LR调度器对象
    """  # noqa: ignore flake8"
    if isinstance(scheduler, str):
        scheduler = scheduler.lower()
        if scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)
        elif scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(optimizer,
                                                          num_warmup_steps=warmup_step)
        elif scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=training_step_num,
            )
        elif scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=training_step_num,
                num_cycles=cosine_schedule_num_cycles,
            )
        elif scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=training_step_num,
                num_cycles=cosine_schedule_num_cycles,
            )
        elif scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_step,
                lr_end=polynomial_decay_schedule_lr_end,
                power=polynomial_decay_schedule_power,
            )
        else:
            raise ValueError('{} is not a valid scheduler'.format(scheduler))

    return scheduler
