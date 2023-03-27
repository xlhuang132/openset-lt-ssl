from collections import defaultdict
import torch

import torch.distributed as dist
import diffdist.functional as distops 


class FeatureQueue:

    def __init__(self, cfg, classwise_max_size=None, bal_queue=False ):
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.feat_dim = cfg.MODEL.QUEUE.FEAT_DIM
        self.max_size = cfg.MODEL.QUEUE.MAX_SIZE
 
        # self.bank = defaultdict(lambda: torch.empty(0, self.feat_dim).cuda())
        # self.prototypes = torch.zeros(self.num_classes, self.feat_dim).cuda()
        
        self.bank = defaultdict(lambda: torch.empty(0, self.feat_dim))
        self.prototypes = torch.zeros(self.num_classes, self.feat_dim)

        self.classwise_max_size = classwise_max_size
        self.bal_queue = bal_queue
        self.only_momentum=False
        if cfg.ALGORITHM.NAME=='MOOD':
            self.only_momentum=True
            self.center_decay_ratio=cfg.MODEL.LOSS.FEATURE_LOSS.CENTER_DECAY_RATIO

    def enqueue(self, features, labels):
        # if self.only_momentum:
        #     for idx in range(self.num_classes): # MOOD
        #         cls_inds = torch.where(labels == idx)[0]
        #         if len(cls_inds):
        #             with torch.no_grad():
        #                 # push to the memory bank
        #                 feats_selected = features[cls_inds]
        #                 mean = feats_selected.mean(dim=0)
        #                 var = feats_selected.var(dim=0, unbiased=False)
        #                 n = feats_selected.numel() / feats_selected.size(1)
        #                 if n > 1:
        #                     var = var * n / (n - 1)
        #                 else:
        #                     var = var 
        #             if torch.count_nonzero(self.prototypes[idx])>0:
        #                 self.prototypes[idx] =(1 - self.center_decay_ratio)* mean    +  \
        #                     self.center_decay_ratio* self.prototypes[idx]
        #             else: 
        #                 self.prototypes[idx] = mean

        # else: # DASO
        # gather_t_1 = [torch.empty_like(features) for _ in range(dist.get_world_size())]
        # gather_t_2 = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        # gather_t_1 = distops.all_gather(gather_t_1, features) #  
        # gather_t_2 = distops.all_gather(gather_t_2, labels) #  
        # features = torch.cat(gather_t_1)
        # labels = torch.cat(gather_t_2) 
        for idx in range(self.num_classes):
            # per class max size
            max_size = (
                self.classwise_max_size[idx] * 5  # 5x samples
            ) if self.classwise_max_size is not None else self.max_size
            if self.bal_queue:
                max_size = self.max_size
            # select features by label
            cls_inds = torch.where(labels == idx)[0]
            if len(cls_inds):
                with torch.no_grad():
                    # push to the memory bank
                    feats_selected = features[cls_inds]
                    self.bank[idx] = torch.cat([self.bank[idx], feats_selected], 0)

                    # fixed size
                    current_size = len(self.bank[idx])
                    if current_size > max_size:
                        self.bank[idx] = self.bank[idx][current_size - max_size:]

                    # update prototypes
                    self.prototypes[idx, :] = self.bank[idx].mean(0)
