
# from .transforms import *
import torchvision
from .transform.rand_augment import RandAugment
from .transform.transforms import TransformFixMatch,TransformOpenMatch
from .transform.transforms import SimCLRAugmentation
from dataset.transform.transforms import Augmentation,GeneralizedSSLTransform
import copy
from .randaugment import RandAugmentMC
from .gaussian_blur import GaussianBlur

from torchvision import transforms

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
    
class TransformTwiceABC:
    def __init__(self, transform,transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3

other_func = {"RandAugmentMC": RandAugmentMC,"GaussianBlur":GaussianBlur}



def get_strong_transform(cfg):
    
    aug = Augmentation 
    dataset=cfg.DATASET.NAME
    resolution = cfg.DATASET.RESOLUTION
    if dataset == "cifar10":
        img_size = (32, 32)
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))

    elif dataset == "cifar100":
        norm_params = dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        img_size = (32, 32)

    elif dataset == "stl10":
        img_size = (96, 96)  # original image size
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    elif dataset =='svhn':
        img_size = (32, 32)  # original image size
        norm_params = dict(mean=(0.4380, 0.4440, 0.4730), std=(0.1751, 0.1771, 0.1744))
    
    strong_transform=aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=False
                )  # strong (randaugment)
           
    return strong_transform
def get_trans(trans_cfg):
    init_params = copy.deepcopy(trans_cfg)
    type_name = init_params.pop("type")
    if type_name in other_func.keys():
        return other_func[type_name](**init_params)
    if type_name == "RandomApply":
        r_trans = []
        trans_list = init_params.pop('transforms')
        for trans_cfg in trans_list:
            r_trans.append(get_trans(trans_cfg))
        return transforms.RandomApply(r_trans, **init_params)

    elif hasattr(transforms, type_name):
        return getattr(transforms, type_name)(**init_params)
    else:
        raise NotImplementedError(
            "Transform {} is unimplemented".format(trans_cfg))
        
class BaseTransform(object):
    """ For torch transform or self write
    """
    def __init__(self, pipeline):
        """ transforms for data
        Args:
            pipelines (list): list of dict, each dict is a transform
        """
        self.pipeline = pipeline
        self.transform = self.init_trans(pipeline)

    def init_trans(self, trans_list):
        trans_funcs = []
        for trans_cfg in trans_list:
            trans_funcs.append(get_trans(trans_cfg))
        return transforms.Compose(trans_funcs)

    def __call__(self, data):
        return self.transform(data)


class ListTransform(BaseTransform):
    """ For torch transform or self write
    """
    def __init__(self, pipelines):
        """ transforms for data
        Args:
            pipelines (list): list of dict, each dict is a transform
        """
        self.pipelines = pipelines
        self.transforms = []
        for trans_dict in self.pipelines:
            self.transforms.append(self.init_trans(trans_dict))

    def __call__(self, data):
        results = []
        for trans in self.transforms:
            results.append(trans(data))
        return results
 
def build_simclr_transform(cfg):
    dataset=cfg.DATASET.NAME
    
    resolution = cfg.DATASET.RESOLUTION
    if dataset == "cifar10":
        img_size = (32, 32)
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))

    elif dataset == "cifar100":
        norm_params = dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        img_size = (32, 32)

    elif dataset == "stl10":
        img_size = (96, 96)  # original image size
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    elif dataset =='svhn':
        img_size = (32, 32)  # original image size
        norm_params = dict(mean=(0.4380, 0.4440, 0.4730), std=(0.1751, 0.1771, 0.1744))
    
    transform=SimCLRAugmentation(cfg, img_size,norm_params=norm_params, resolution=resolution)
    return TransformTwice(transform)

def build_transform(cfg):
    
    algo_name = cfg.ALGORITHM.NAME 
    
    resolution = cfg.DATASET.RESOLUTION
    
    dataset=cfg.DATASET.NAME
    aug = Augmentation 
     
    if dataset == "cifar10":
        img_size = (32, 32)
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))

    elif dataset == "cifar100":
        norm_params = dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        img_size = (32, 32)

    elif dataset == "stl10":
        img_size = (96, 96)  # original image size
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    elif dataset =='svhn':
        img_size = (32, 32)  # original image size
        norm_params = dict(mean=(0.4380, 0.4440, 0.4730), std=(0.1751, 0.1771, 0.1744))
    
    l_train = aug(
            cfg, img_size, 
            strong_aug=cfg.DATASET.TRANSFORM.LABELED_STRONG_AUG, 
            norm_params=norm_params, 
            resolution=resolution
        ) 
    
    if algo_name == "MixMatch":
        # K weak
        ul_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution)
                for _ in range(cfg.ALGORITHM.MIXMATCH.NUM_AUG)
            ]
        )

    elif algo_name in ['ReMix']:
        # 1 weak + K strong
        augmentations = [aug(cfg, img_size, norm_params=norm_params, resolution=resolution)]
        for _ in range(cfg.ALGORITHM.REMIXMATCH.NUM_AUG):
            augmentations.append(
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=False
                )
            )
        ul_train = GeneralizedSSLTransform(augmentations)

    elif algo_name == "USADTM":
        # identity + weak + strong
        ul_train = GeneralizedSSLTransform(
            [
                aug(
                    cfg,
                    img_size,
                    norm_params=norm_params,
                    resolution=resolution,
                    flip=False,
                    crop=False
                ),  # identity
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),  # weak
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                )  # strong (randaugment)
            ]
        )

    elif algo_name == "PseudoLabel":
        # 1 weak
        ul_train = GeneralizedSSLTransform(
            [aug(cfg, img_size, norm_params=norm_params, resolution=resolution)]
        ) 
    elif algo_name=='OpenMatch':
        l_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),  # weak
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                ),  # strong (randaugment)
                aug(
                    cfg,
                    img_size,
                    norm_params=norm_params,
                    resolution=resolution,
                    flip=False,
                    crop=False
                ),  # identity
            ]
        )
        ul_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),  # weak
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                ),  # strong (randaugment)
                aug(
                    cfg,
                    img_size,
                    norm_params=norm_params,
                    resolution=resolution,
                    flip=False,
                    crop=False
                ),  # identity
            ]
        )
    elif algo_name in ['CCSSL','OODDetect','DCSSL','ABC','CoSSL']: #'DCSSL'
        ul_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),  # weak
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                ),  # strong (randaugment)
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=False
                ),  # strong (randaugment)
            ]
        )
    # elif algo_name=='DCSSL':
    #     ul_train = GeneralizedSSLTransform(
    #         [
    #             aug(cfg, img_size, norm_params=norm_params, resolution=resolution),  # weak
    #             aug(
    #                 cfg,
    #                 img_size,
    #                 strong_aug=True,
    #                 norm_params=norm_params,
    #                 resolution=resolution,
    #                 ra_first=True
    #             ),  # strong (randaugment)
    #             aug(
    #                 cfg,
    #                 img_size,
    #                 strong_aug=True,
    #                 norm_params=norm_params,
    #                 resolution=resolution,
    #                 ra_first=False
    #             ),  # strong (randaugment)
    #         ]
    #     )
    #     l_train=GeneralizedSSLTransform(
    #         [
    #             aug(
    #         cfg, img_size, 
    #         strong_aug=cfg.DATASET.TRANSFORM.LABELED_STRONG_AUG, 
    #         norm_params=norm_params, 
    #         resolution=resolution
    #     ) ,
    #          aug(
    #                 cfg,
    #                 img_size,
    #                 strong_aug=True,
    #                 norm_params=norm_params,
    #                 resolution=resolution,
    #                 ra_first=True
    #             ),  # strong (randaugment)
    #             aug(
    #                 cfg,
    #                 img_size,
    #                 strong_aug=True,
    #                 norm_params=norm_params,
    #                 resolution=resolution,
    #                 ra_first=False
    #             ),  # strong (randaugment)
    #         ]
    #     )
    else:
        ul_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),
                aug(
                    cfg,
                    img_size,
                    strong_aug=cfg.DATASET.TRANSFORM.UNLABELED_STRONG_AUG,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                )
            ]
        )
    eval_aug = Augmentation(
        cfg,
        img_size,
        flip=False,
        crop=False,
        norm_params=norm_params,
        is_train=False,
        resolution=resolution
    )
    return l_train,ul_train,eval_aug