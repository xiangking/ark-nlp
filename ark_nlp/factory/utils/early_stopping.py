import copy
import torch
import numpy as np


class EarlyStopping:

    # https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook

    def __init__(self, patience=2, mode="max", epoch_num=10, min_epoch=0, at_last_score=None, **kwargs):
        self.patience = patience
        self.mode = mode
        self.max_epoch = epoch_num
        self.min_epoch = min_epoch
        self.at_last_score = at_last_score if at_last_score is not None else -np.Inf
        self.epoch = 0
        self.early_stop = False
        self.best_model = None
        self.best_epoch = 0
        self.model_path = None
        self.best_score = -np.Inf if self.mode == "max" else np.Inf

    def __call__(self, epoch_score, model=None, model_path=None):
        self.model_path = model_path
        self.epoch += 1

        score = -epoch_score if self.mode == "min" else epoch_score

        if score <= self.best_score:
            counter = self.epoch - self.best_epoch
            print('EarlyStopping counter: {} out of {}'.format(counter, self.patience))
            if (counter >= self.patience) and (self.best_score > self.at_last_score) and (self.epoch >= self.min_epoch):
                self.early_stop = True
                self._save_checkpoint()
        else:
            self.best_score = score
            self.best_epoch = self.epoch
            self.best_model = copy.deepcopy(model.state_dict())
        print('best_score is:{}'.format(self.best_score))

        if self.max_epoch <= self.epoch:
            self.early_stop = True
            self._save_checkpoint()

    def _save_checkpoint(self):
        if self.model_path is not None and self.best_model is not None:
            torch.save(self.best_model, self.model_path.replace('_score', '_' + str(self.best_score)))
            print('model saved at: ', self.model_path.replace('_score', '_' + str(self.best_score)))