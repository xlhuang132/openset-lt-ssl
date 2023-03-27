import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        with torch.no_grad(): 
            p = F.softmax(logits, dim=1)
            if np.isnan(p.mean().item()):
                raise NameError('GCE_p')
            # modify gradient of cross entropy
            Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
            loss_weight = (Yg.squeeze().detach()**self.q)*self.q
            if np.isnan(Yg.mean().item()):
                raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight

        return loss