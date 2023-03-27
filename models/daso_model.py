from .resnet_multi_head import ResNet_s_mul_h 
from .feature_queue import FeatureQueue 
class DASOModel(ResNet_s_mul_h):
    def __init__(): 
        super(ResNet_s_mul_h, self).__init__(cfg)
        self.queue = FeatureQueue(cfg, classwise_max_size=None, bal_queue=True)
        self.conf_thres = cfg.ALGORITHM.CONFIDENCE_THRESHOLD

        self.similarity_fn = nn.CosineSimilarity(dim=2)

        self.pretraining = True
        self.T_proto = cfg.ALGORITHM.DASO.PROTO_TEMP
        self.pretraining = True
        self.psa_loss_weight = cfg.ALGORITHM.DASO.PSA_LOSS_WEIGHT

        self.T_dist = cfg.ALGORITHM.DASO.DIST_TEMP
        self.with_dist_aware = cfg.ALGORITHM.DASO.WITH_DIST_AWARE
        self.interp_alpha = cfg.ALGORITHM.DASO.INTERP_ALPHA

        self.cfg = cfg

        # logit adjustment
        self.with_la = cfg.ALGORITHM.LOGIT_ADJUST.APPLY  # train la
        self.tau = cfg.ALGORITHM.LOGIT_ADJUST.TAU

        self.num_classes = cfg.MODEL.NUM_CLASSES
        
    def encoder(self,x,**kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        encoding = out.view(out.size(0), -1)  
        return encoding
         
        
    def classifier(self,x):
        return self.fc(x)

    def forward(self, x,training=True,return_features_only=False, **kwargs):
        encoding=self.encoder(x)
        if return_features_only:
            return self.encoder(x)
        return self.fc(encoding)
        

    