# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)

use_cuda=False



# device
def __setup_cuda(to_use_cuda:bool = True):
    global use_cuda
    if to_use_cuda and torch.cuda.is_available():
        use_cuda = True
        torch.defaultDevice=torch.device('cuda:0')
    else:
        use_cuda = False
        torch.defaultDevice=torch.device('cpu')
    torch.set_default_device(torch.defaultDevice)
    return


# precision
def __set_default_float(precision=32):
    """Sets the default float type.
    """
    if precision == 32:
        np.defaultReal=np.float32
        torch.defaultReal=torch.float32
    elif precision == 64:
        np.defaultReal=np.float64
        torch.defaultReal=torch.float64
    else:
        raise Exception(f"unknown float type {precision}")
    torch.set_default_dtype(torch.defaultReal)
    return

# reproductivity
def __set_random_seed(seed=None):
    """Sets all random seeds for Python random, NumPy, and torch
    """
    if seed is not None:
        random.seed(seed)  # python random
        np.random.seed(seed)  # numpy
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def oneKey_configure(use_cuda:bool =None,
                     precision:int=None,
                     seed:int=None):
    if use_cuda is not None:
        __setup_cuda(to_use_cuda=use_cuda)
    if precision is not None:
        __set_default_float(precision=precision)
    if seed is not None:
        __set_random_seed(seed=seed)
    logger.info(f"""
          Configured successfully:
          Device:{torch.defaultDevice}, numpy.dtype: {np.defaultReal}, tensor.dtype: {torch.defaultReal}
          """)

#oneKey_configure(use_cuda=True, precision=32)
oneKey_configure(use_cuda=True, precision=64, seed=42)
