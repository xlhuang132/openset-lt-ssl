
from .base_trainer import BaseTrainer
from .daso_trainer import DASOTrainer
from .mood_trainer import MOODTrainer
from .fixmatch_trainer import FixMatchTrainer
from .mixmatch_trainer import MixMatchTrainer
from .crest_trainer import CReSTTrainer
from .supervised_trainer import SupervisedTrainer
from .pseudolabel_trainer import PseudoLabelTrainer
from .openmatch_trainer import OpenMatchTrainer
from .dcssl_trainer import DCSSLTrainer
from .ccssl_trainer import CCSSLTrainer
from .mtcf_trainer import MTCFTrainer
from .acssl_trainer import ACSSLTrainer
def build_trainer(cfg):
    alg=cfg.ALGORITHM.NAME
    if alg=='MOOD':
        return MOODTrainer(cfg)
    elif alg=='baseline':
        return SupervisedTrainer(cfg)
    elif alg=='FixMatch':
        return FixMatchTrainer(cfg)
    elif alg=='MixMatch':
        return MixMatchTrainer(cfg)
    elif alg=='CReST':
        return CReSTTrainer(cfg)
    elif alg=='DASO':
        return DASOTrainer(cfg)
    elif alg=='PseudoLabel':
        return PseudoLabelTrainer(cfg)
    elif alg=='OpenMatch':
        return OpenMatchTrainer(cfg)
    elif alg=='MTCF':
        return MTCFTrainer(cfg)
    elif alg== 'DCSSL':
        return DCSSLTrainer(cfg)    
    elif alg== 'CCSSL':
        return CCSSLTrainer(cfg) 
    elif alg== 'ACSSL':
        return ACSSLTrainer(cfg) 
    
    else:
        raise "The algorithm type is not valid!"