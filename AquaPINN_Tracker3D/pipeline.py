# -*- coding: utf-8 -*-
import os
from logger import getLogger
import config
from RawDataProcess import decoder 
from Sync import syncer
from Track import tracker
from SharedLibs.functions import  load_params
from system import localizationSystem
import json


# param_file='demos\SyntheticData\params_synthetic.json'
#param_file='demos\SyntheticData2\params_synthetic2.json'
param_file='..\demos\LevyPark\params_levy_park.json'
#param_file='..\demos\McCloud_2024_study\params_McCloud.json'
# param_file='demos\LGS18_tailrace_north\params_LGS18.json'
# param_file='demos\York_Haven\params_York.json'
# param_file='demos\Levy_Park_forTest\params_levy_park.json'
#param_file='demos\LGS18_tailrace_north_forTest\params_LGS18.json'
# param_file=r'..\demos\Profish\params_Profish.json'
param_file =r'..\demos\fishHeart_test\params_fishHeart.json'
# param_file=r'..\demos\Ice_Harbor\params_IHR.json'
# param_file=r'..\demos\Ice_Harbor112524_test\params_IHR.json'
# param_file=r'..\demos\YorkHaven112624\params_York3D.json'
# param_file=r'..\demos\SwanFalls_2024\params_SwanFalls.json'
# param_file=r'..\demos\SwanFalls_2024Release\params_SwanFalls.json'
# param_file=r'..\demos\YorkHaven101124\params_York.json'
param_file=r'..\demos\YorkHaven100924\params_York.json'
# param_file=r'..\demos\LMN23_EA\params_LMN.json'
# param_file=r'..\demos\YorkHaven2024Release\params_York.json'

doSync=False
trackBeacon=False

config.oneKey_configure(use_cuda=True, precision=64, seed=42)

train_params= load_params(param_file)[0]
       
TagStr = "Tag"+str(train_params["TagID"])
logger=getLogger(logFile=os.path.join(train_params["output_folder"], f'history_{TagStr}.log') )

loc_sys = localizationSystem().deploy(param_file, doSync=doSync)


decoder.run(param_file,loc_sys)    
if doSync:
    syncer.run(param_file,loc_sys)
    pass 

if trackBeacon:
    param_file_beacon=os.path.splitext(param_file)[0]+'_Beacon.json'
    train_params['TagID']=[-1]
    for k,v in train_params.items():
        if k == 'TagID': continue
        if k in ['GPSFile', 'testFile']: 
            train_params[k]=[""]
            continue
        train_params[k]=[v]
    with open(param_file_beacon, 'w') as f:
        json.dump(train_params, f, sort_keys=False, indent=4)    
    tracker.run(param_file_beacon,loc_sys,trackBeacon=True)

##
tracker.run(param_file,loc_sys)

