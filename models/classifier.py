# import torch.nn as nn
# from torch import Tensor


# class Classifier(nn.Module):

#     def __init__( self, in_features: int, out_features: int, *, bias: bool = True
#     ):
#         super(Classifier,self).__init__()
#         self.fc = nn.Linear(in_features, out_features, bias=bias)
#         self._init_weights()

#     def forward(self, x):
#         return self.fc(x)

#     def _init_weights(self):
#         for m in self.fc.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 nn.init.constant_(m.bias, 0)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.in_features=in_features 
        self.fc = nn.Linear(in_features, num_classes) 
        self._init_weights()

    def _init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
            
        output=self.fc(x)

        return output 