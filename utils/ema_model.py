import copy 
import torch.nn as nn

def create_ema_model(model):
        ema_model = copy.deepcopy(model) 
 
        for param in ema_model.parameters():
            param.detach_()

        return ema_model

class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999, wd=True):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        if wd:
            self.wd = 0.02 * lr
        else:
            self.wd = 0.0

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

