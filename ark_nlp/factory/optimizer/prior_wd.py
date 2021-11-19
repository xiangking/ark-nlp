from torch.optim import Optimizer


class PriorWD(Optimizer):
    def __init__(self, optim, use_prior_wd=False, exclude_last_group=True):
        super(PriorWD, self).__init__(optim.param_groups, optim.defaults)

        # python dictionary does not copy by default
        self.param_groups = optim.param_groups
        self.optim = optim
        self.use_prior_wd = use_prior_wd
        self.exclude_last_group = exclude_last_group

        self.weight_decay_by_group = []
        for i, group in enumerate(self.param_groups):
            self.weight_decay_by_group.append(group["weight_decay"])
            group["weight_decay"] = 0

        self.prior_params = {}
        for i, group in enumerate(self.param_groups):
            for p in group["params"]:
                self.prior_params[id(p)] = p.detach().clone()

    def step(self, closure=None):
        if self.use_prior_wd:
            for i, group in enumerate(self.param_groups):
                for p in group["params"]:
                    if self.exclude_last_group and i == len(self.param_groups):
                        p.data.add_(-group["lr"] * self.weight_decay_by_group[i], p.data)
                    else:
                        p.data.add_(
                            -group["lr"] * self.weight_decay_by_group[i], p.data - self.prior_params[id(p)],
                        )
        loss = self.optim.step(closure)

        return loss

    def compute_distance_to_prior(self, param):
        """
        Compute the L2-norm between the current parameter value to its initial (pre-trained) value.
        """
        assert id(param) in self.prior_params, "parameter not in PriorWD optimizer"
        return (param.data - self.prior_params[id(param)]).pow(2).sum().sqrt()
