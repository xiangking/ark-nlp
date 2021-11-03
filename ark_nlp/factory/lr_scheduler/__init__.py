from transformers import get_linear_schedule_with_warmup


def get_default_linear_schedule_with_warmup(
    optimizer,
    t_total,
    warmup_ratio: float = 0.06,
):
    '''
    the learning rate scheduler.
    '''
    warmup_steps = int(t_total * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total
    )
    return scheduler
