# -*- coding: utf-8 -*-

from .decorators import treeFunction
import numpy as np
import torch
import json
import itertools
import os
try:
    import tomllib           # Python >= 3.11 (stdlib)
except ImportError:
    import tomli as tomllib  # pip install tomli

treeToNumpy = treeFunction(lambda x: x.detach().cpu().numpy()
                           if not isinstance(x,np.ndarray)
                           else x)
treeToTensor= treeFunction(lambda x:
                           torch.as_tensor(x, dtype=torch.defaultReal,
                           device=torch.defaultDevice))

def load_params(param_file):
	params = None
	if param_file.endswith('.toml'):
		with open(param_file, 'rb') as f:
			params = tomllib.load(f)
	else:
		with open(param_file, 'r') as f:
			params = json.load(f)
    # the combinations of parameters
	allNames = sorted(params)
	combinations = itertools.product(*(params[Name] for Name in allNames))
	combo_list = list(combinations)
	train_params_list = []
	for combo in combo_list:
		train_params = {}
		for n in range(len(allNames)):
			name = allNames[n]
			train_params[name] = combo[n]
		train_params_list.append(train_params)
	return train_params_list

def soundSpeed_freshWater(T):
    sound_speed = (  1.402385E3 + 5.038813* T -5.799136E-2 * T**2 
                    +3.287156E-4 * T**3  - 1.398845E-6* T**4 + 2.78786E-9* T**5 )
    return sound_speed      
  

def replaceFolder(filePath, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(folder,os.path.basename(filePath))