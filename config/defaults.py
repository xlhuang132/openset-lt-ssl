from yacs.config import CfgNode as CN

_C = CN()
_C.LOCAL_RANK=0
_C.FINETUNE_STEP=50000
_C.MISC = CN()
_C.MISC.LOG_CLASSWISE = True

# Model
_C.MODEL = CN()
_C.MODEL.NAME = "WRN"
_C.MODEL.WIDTH = 2
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.EMA_DECAY = 0.999
_C.MODEL.EMA_WEIGHT_DECAY = 0.0
_C.MODEL.WITH_ROTATION_HEAD = False
_C.MODEL.DUAL_HEAD_ENABLE=False
_C.MODEL.DUAL_HEAD_OUT_DIM=0
_C.MODEL.PROJECT_FEATURE_DIM=128

# Feature Queue for DASO and USADTM
_C.MODEL.QUEUE = CN()
_C.MODEL.QUEUE.MAX_SIZE = 256
_C.MODEL.QUEUE.FEAT_DIM = 128

# Losses
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.LABELED_LOSS = "CrossEntropyLoss"
_C.MODEL.LOSS.LABELED_LOSS_CLASS_WEIGHT_TYPE= "None"
_C.MODEL.LOSS.WARMUP_LABELED_LOSS_CLASS_WEIGHT_TYPE="None"
_C.MODEL.LOSS.WITH_LABELED_COST_SENSITIVE = False
_C.MODEL.LOSS.FEATURE_LOSS=CN()
_C.MODEL.LOSS.FEATURE_LOSS.TEMPERATURE=1.
_C.MODEL.LOSS.FEATURE_LOSS.WEIGHT=1.
_C.MODEL.LOSS.FEATURE_LOSS.CENTER_DECAY_RATIO=0.1 
_C.MODEL.LOSS.FEATURE_LOSS.OOD_LOSS_WEIGHT=1.
_C.MODEL.LOSS.FEATURE_LOSS.ID_LOSS_WEIGHT=1.

_C.MODEL.LOSS.UNLABELED_LOSS = "CrossEntropyLoss"
_C.MODEL.LOSS.UNLABELED_LOSS_WEIGHT = 1.0  #
_C.MODEL.LOSS.WITH_SUPPRESSED_CONSISTENCY = False 

# _C.MODEL.LOSS.COST_SENSITIVE = CN()
# _C.MODEL.LOSS.COST_SENSITIVE.LOSS_OVERRIDE = ""  # default: balanced CE loss (CB, LDAM)
# _C.MODEL.LOSS.COST_SENSITIVE.BETA = 0.999

# Cross Entropy
_C.MODEL.LOSS.CROSSENTROPY = CN()
_C.MODEL.LOSS.CROSSENTROPY.USE_SIGMOID = False

_C.MODEL.OPTIMIZER = CN()
_C.MODEL.OPTIMIZER.TYPE = "SGD"
_C.MODEL.OPTIMIZER.BASE_LR = 0.03
_C.MODEL.OPTIMIZER.MOMENTUM = 0.9
_C.MODEL.OPTIMIZER.WEIGHT_DECAY = 1e-4


_C.MODEL.LR_SCHEDULER = CN()
_C.MODEL.LR_SCHEDULER.TYPE = "multistep"
_C.MODEL.LR_SCHEDULER.LR_STEP = [40, 50]
_C.MODEL.LR_SCHEDULER.LR_FACTOR = 0.1
_C.MODEL.LR_SCHEDULER.WARM_EPOCH = 5
_C.MODEL.LR_SCHEDULER.COSINE_DECAY_END = 0
 

# Algorithm
_C.ALGORITHM = CN()
_C.ALGORITHM.ABLATION=CN()
_C.ALGORITHM.ABLATION.ENABLE=False
_C.ALGORITHM.ABLATION.DUAL_BRANCH=False 
_C.ALGORITHM.ABLATION.MIXUP= False
_C.ALGORITHM.ABLATION.OOD_DETECTION= False
_C.ALGORITHM.ABLATION.PAP_LOSS= False 


_C.ALGORITHM.NAME = "Supervised"
_C.ALGORITHM.CONFIDENCE_THRESHOLD = 0.95
_C.ALGORITHM.CONS_RAMPUP_SCHEDULE = "exp"  # "exp" or "linear"
_C.ALGORITHM.CONS_RAMPUP_ITERS_RATIO = 0.4
# 

_C.ALGORITHM.PRE_TRAIN=CN()
_C.ALGORITHM.PRE_TRAIN.ENABLE=False
_C.ALGORITHM.PRE_TRAIN.WARMUP_EPOCH=0
_C.ALGORITHM.PRE_TRAIN.SimCLR=CN()
_C.ALGORITHM.PRE_TRAIN.SimCLR.ENABLE=False
_C.ALGORITHM.PRE_TRAIN.SimCLR.K=200
_C.ALGORITHM.PRE_TRAIN.SimCLR.TEMPERATURE=0.5
_C.ALGORITHM.PRE_TRAIN.SimCLR.FEATURE_DIM=64

_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR=CN()
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.TEMPERATURE=1. # KNN OOD DETECTOR 加权
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.MAGNITUDE=0.001  # 输入扰动尺度
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.THRESHOLD=0.95  # 阈值 
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.K=10  # top-k
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.DETECT_EPOCH=30
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.DOMAIN_Y_UPDATE_ITER=5 
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.FEATURE_DECAY_RATIO = 0.1  #  
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.OOD_THRESHOLD=0.5 # top-k低于这个比率就是OOD
_C.ALGORITHM.PRE_TRAIN.OOD_DETECTOR.ID_THRESHOLD=0.5 # top-k超过这个比率就是ID


_C.ALGORITHM.BRANCH1_MIXUP= False # 是否对第1个采样通道进行mixup
_C.ALGORITHM.BRANCH2_MIXUP= False # 是否对第1个采样通道进行mixup

# # PseudoLabel
_C.ALGORITHM.PSEUDO_LABEL = CN()

# # Mean Teacher
_C.ALGORITHM.MEANTEACHER = CN()
_C.ALGORITHM.MEANTEACHER.APPLY_DASO = False

# # MixMatch
_C.ALGORITHM.MIXMATCH = CN()
_C.ALGORITHM.MIXMATCH.NUM_AUG = 2
_C.ALGORITHM.MIXMATCH.TEMPERATURE = 0.5
_C.ALGORITHM.MIXMATCH.MIXUP_ALPHA = 0.75
_C.ALGORITHM.MIXMATCH.APPLY_DASO = False

# # ReMixMatch
_C.ALGORITHM.REMIXMATCH = CN()
_C.ALGORITHM.REMIXMATCH.NUM_AUG = 2
_C.ALGORITHM.REMIXMATCH.TEMPERATURE = 0.5
_C.ALGORITHM.REMIXMATCH.MIXUP_ALPHA = 0.75
_C.ALGORITHM.REMIXMATCH.WEIGHT_KL = 1.0
_C.ALGORITHM.REMIXMATCH.WEIGHT_ROT = 1.0
_C.ALGORITHM.REMIXMATCH.WITH_DISTRIBUTION_MATCHING = True
_C.ALGORITHM.REMIXMATCH.LABELED_STRONG_AUG = False

# # FixMatch
_C.ALGORITHM.FIXMATCH = CN()
# # FixMatchBCL
_C.ALGORITHM.FIXMATCHBCL = CN()
_C.ALGORITHM.FIXMATCHBCL.TEMPERATURE=0.07 
_C.ALGORITHM.FIXMATCHBCL.LAMBDA_BCL=0.35
# DASO
_C.ALGORITHM.DASO = CN()
# _C.ALGORITHM.DASO.PRETRAIN_STEPS = 5000
_C.ALGORITHM.DASO.WARMUP_EPOCH=10
_C.ALGORITHM.DASO.PROTO_TEMP = 0.05
_C.ALGORITHM.DASO.PL_DIST_UPDATE_PERIOD = 100

# pseudo-label mixup
_C.ALGORITHM.DASO.WITH_DIST_AWARE = True
_C.ALGORITHM.DASO.DIST_TEMP = 1.0
_C.ALGORITHM.DASO.INTERP_ALPHA = 0.5
_C.ALGORITHM.DASO.WARMUP_ITER=5000
# prototype option
_C.ALGORITHM.DASO.QUEUE_SIZE = 256

# Semantic Alignment loss
_C.ALGORITHM.DASO.PSA_LOSS_WEIGHT = 1.0

# # CReST
_C.ALGORITHM.CREST = CN()
_C.ALGORITHM.CREST.GEN_PERIOD_EPOCH = 50  # MAX_EPOCH/GEN_PEROID_EPOCH = N gen
_C.ALGORITHM.CREST.ALPHA = 3.0
_C.ALGORITHM.CREST.TMIN = 0.5
_C.ALGORITHM.CREST.PROGRESSIVE_ALIGN = False

# # ABC
_C.ALGORITHM.ABC = CN()
_C.ALGORITHM.ABC.APPLY = False
_C.ALGORITHM.ABC.DASO_PSEUDO_LABEL = True
 
# cRT
_C.ALGORITHM.CRT = CN()
_C.ALGORITHM.CRT.TARGET_DIR = ""
 
 
# OpenCos
_C.ALGORITHM.OPENCOS=CN() 
_C.ALGORITHM.OPENCOS.THS=1.
_C.ALGORITHM.OPENCOS.TEMP_S2=1. # temperature scaling
_C.ALGORITHM.OPENCOS.LMD_UNIF=1. # smoothing loss weight
_C.ALGORITHM.OPENCOS.LAMBDA_U=1.

# OpenMatch
_C.ALGORITHM.OPENMATCH=CN() 
_C.ALGORITHM.OPENMATCH.LAMBDA_OEM=1.
_C.ALGORITHM.OPENMATCH.LAMBDA_SOCR=1. # temperature scaling
_C.ALGORITHM.OPENMATCH.MU=2. # smoothing loss weight 
_C.ALGORITHM.OPENMATCH.START_FIX=10
_C.ALGORITHM.OPENMATCH.T=1.

# MOOD
_C.ALGORITHM.MOOD = CN()
_C.ALGORITHM.MOOD.NUM_AUG = 2
_C.ALGORITHM.MOOD.TEMPERATURE = 0.5
_C.ALGORITHM.MOOD.MIXUP_ALPHA = 0.75
_C.ALGORITHM.MOOD.BETA = 0.999 
_C.ALGORITHM.MOOD.FEATURE_LOSS_TEMPERATURE=1.
_C.ALGORITHM.MOOD.PAP_LOSS_WEIGHT=1.

# DCSSL
_C.ALGORITHM.DCSSL = CN() 
_C.ALGORITHM.DCSSL.LAMBDA_D = 1.   
_C.ALGORITHM.DCSSL.DCSSL_CONTRA_TEMPERTURE=0.07
_C.ALGORITHM.DCSSL.FP_K=1
_C.ALGORITHM.DCSSL.M=0.999
_C.ALGORITHM.DCSSL.CONTRAST_WITH_THRESH=False
_C.ALGORITHM.DCSSL.CONTRAST_WITH_P_DIASED=False
_C.ALGORITHM.DCSSL.CONTRAST_THRESH=0.9
_C.ALGORITHM.DCSSL.CONTRAST_WITH_HP=False
_C.ALGORITHM.DCSSL.CONTRAST_WITH_HN=False
_C.ALGORITHM.DCSSL.LOSS_VERSION=1
# ACSSL
_C.ALGORITHM.ACSSL = CN() 
_C.ALGORITHM.ACSSL.LAMBDA_A = 1.   
_C.ALGORITHM.ACSSL.T1=0.04
_C.ALGORITHM.ACSSL.T2=0.1
_C.ALGORITHM.ACSSL.HEAD_NUM=8
 
# RCSSL
_C.ALGORITHM.RESSL = CN()
_C.ALGORITHM.RESSL.TEMP_K = 0.04
_C.ALGORITHM.RESSL.TEMP_Q = 0.1
_C.ALGORITHM.RESSL.K=65536
_C.ALGORITHM.RESSL.M=0.999
_C.ALGORITHM.RESSL.LAMBDA_R=1.

# CCSSL
_C.ALGORITHM.CCSSL = CN()
_C.ALGORITHM.CCSSL.U_T = 1.
_C.ALGORITHM.CCSSL.CONTRAST_THRESH=0.9
_C.ALGORITHM.CCSSL.TEMPERATURE=0.07
_C.ALGORITHM.CCSSL.LAMBDA_C=1. 
_C.ALGORITHM.CCSSL.CONTRAST_LEFT_OUT=False
_C.ALGORITHM.CCSSL.CONTRAST_WITH_SOFTLABEL=False
_C.ALGORITHM.CCSSL.CONTRAST_WITH_THRESH=False
# MTCF

_C.ALGORITHM.MTCF = CN()
_C.ALGORITHM.MTCF.MIXUP_ALPHA = 0.75 
_C.ALGORITHM.MTCF.LAMBDA_U=75
_C.ALGORITHM.MTCF.T=0.5 # 多少次幂  

# dataset
_C.DATASET = CN()
_C.DATASET.BUILDER = "build_cifar10_dataset"
_C.DATASET.NAME = "cifar10" # 需要更改
_C.DATASET.ROOT = "./data"

_C.DATASET.IMAGENET=CN()
_C.DATASET.IMAGENET.PERCENT=0.1 

#  
_C.DATASET.SAMPLER=CN()
_C.DATASET.SAMPLER.NAME = "RandomSampler"
_C.DATASET.SAMPLER.BETA = 0.999
# 添加双采样分支
_C.DATASET.DUAL_SAMPLER=CN()
_C.DATASET.DUAL_SAMPLER.ENABLE=False
_C.DATASET.DUAL_SAMPLER.NAME="RandomSampler"

_C.DATASET.RESOLUTION = 32
_C.DATASET.BATCH_SIZE = 64 
_C.DATASET.NUM_WORKERS = 8
_C.DATASET.DOMAIN_DATASET_RETURN_INDEX=False
_C.DATASET.UNLABELED_DATASET_RETURN_INDEX=False
_C.DATASET.LABELED_DATASET_RETURN_INDEX=False
_C.DATASET.DL=CN()

_C.DATASET.GROUP_SPLITS=[3,3,4]
_C.DATASET.IFS=[100]
_C.DATASET.OODRS=[0.0]
 
_C.DATASET.IMB_TYPE='exp'

_C.DATASET.DL.NUM_LABELED_HEAD = 1500
_C.DATASET.DL.IMB_FACTOR_L = 100

_C.DATASET.DU=CN()
_C.DATASET.DU.TOTAL_NUM=10000
_C.DATASET.DU.UNLABELED_BATCH_RATIO=2
_C.DATASET.DU.ID=CN()
_C.DATASET.DU.ID.NUM_UNLABELED_HEAD = 3000
_C.DATASET.DU.ID.IMB_FACTOR_UL = 100
_C.DATASET.DU.ID.REVERSE_UL_DISTRIBUTION = False

_C.DATASET.DU.OOD=CN()
_C.DATASET.DU.OOD.ENABLE=False
_C.DATASET.DU.OOD.DATASET=''
_C.DATASET.DU.OOD.ROOT='./data'
_C.DATASET.DU.OOD.INCLUDE_ALL=False
_C.DATASET.DU.OOD.RATIO=0.0 # 
_C.DATASET.NUM_CLASSES=10
_C.SAVE_EPOCH=100
_C.MAX_EPOCH=500 
_C.VAL_ITERATION=500
_C.SHOW_STEP=20
_C.TRAIN_STEP=100 
 
# transform parameters
_C.DATASET.TRANSFORM = CN()
_C.DATASET.TRANSFORM.UNLABELED_STRONG_AUG = True
_C.DATASET.TRANSFORM.LABELED_STRONG_AUG = False
_C.DATASET.TRANSFORM.STRONG_AUG = True

_C.DATASET.NUM_VALID= 0 #   5000
_C.DATASET.REVERSE_UL_DISTRIBUTION=False
# analyse model 

_C.ANALYSE_TYPE= [] 

_C.OUTPUT_DIR = "outputs"
_C.RESUME = '' 
_C.EVAL_ON_TEST_SET = True
_C.GPU_ID = 0

# Reproducability
_C.SEED = 7
_C.SEED_PATH_ENABLE=False
_C.CUDNN_DETERMINISTIC = True
_C.CUDNN_BENCHMARK = False

_C.GPU_MODE=True
def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
