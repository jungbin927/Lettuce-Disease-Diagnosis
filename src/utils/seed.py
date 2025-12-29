import torch
import numpy as np 
import random


def set_seed(seed: int = 42):
    """재현성 유지를 위해 난수 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True