import numpy as np
import os
import torch
import random

def set_seed(seed=1029):
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	np.random.seed(seed)  # Numpy module.
	random.seed(seed)  # Python random module.
	torch.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True   