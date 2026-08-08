# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:21:41 2024

@author: chen096
"""

import logging
logger = logging.getLogger(__name__)
import os
import pandas as pd
#import torch,sys
from ..file import fileSystem
from ..SharedLibs.functions import load_params
from ..SharedLibs.decorators import timing
from .receiver_dataprep import compile_tagdata2dict, cleanCache
from ..SharedLibs.visualization import showDetections


@timing
def run(param_file, loc_sys):
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('+++++++++++++++++++ DECODE ... ++++++++++++++++++++')
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    phoneInfo, tagInfo=loc_sys.phoneInfo, loc_sys.tagInfo
    # load parameters
    train_params= load_params(param_file)[0]
    time_start_UTC=pd.to_datetime(train_params.get('time_start', None))-pd.Timedelta(hours=loc_sys.timeZone_local)    
    time_end_UTC=pd.to_datetime(train_params.get('time_end', None))-pd.Timedelta(hours=loc_sys.timeZone_local)
    fartherPath=train_params['output_folder']
    detectionOutputFolder=os.path.join(fartherPath, 'detections')
    if not os.path.isdir(detectionOutputFolder):
        os.mkdir(detectionOutputFolder)
    # start model training process
    datafolder  = train_params['dataFolder']
    #file with tag code
    tagFile = train_params['tagFile']
    decodesFile = loc_sys.getDecodesFile(tagFile,'Tag', newFolder=fartherPath)
    if os.path.isfile(decodesFile):
        logger.info(f"File already exist: {decodesFile}")
    else:
        compile_tagdata2dict(loc_sys,tagInfo, datafolder,save=True,filename=decodesFile, time_start_UTC=time_start_UTC, time_end_UTC=time_end_UTC)        
        showDetections(decodesFile, tagType='Tag', 
                       outputdir=detectionOutputFolder, timeZone=loc_sys.timeZone_local)
        
    decodesFile_beacons = loc_sys.getDecodesFile(tagFile,'Beacon', newFolder=fartherPath)
    BeaconInfo = {'code':phoneInfo['AttachedBeacons']}
    if BeaconInfo is not None:
        if os.path.isfile(decodesFile_beacons):
            logger.info(f"File already exist: {decodesFile_beacons}")
        else:        
            compile_tagdata2dict(loc_sys,BeaconInfo, datafolder,save=True,filename=decodesFile_beacons, time_start_UTC=time_start_UTC, time_end_UTC=time_end_UTC)     
            showDetections(decodesFile_beacons, tagType='Beacon', 
                           outputdir=detectionOutputFolder, timeZone=loc_sys.timeZone_local)
    cleanCache()
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('+++++++++++++++++++ DECODE done ++++++++++++++++++++')
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')   


if __name__=='__main__':
    # shad tag EA
    # param_file   = r'Shad_EA/params_shad.json'
    # param_file   = r'Shad_beacons/params_shad_beacons.json'
    # param_file = r'LMN23_dam_beacons/params_LMN23_EA.json'
    param_file = r'params.json'
    # param_file = r'LMN23_EA/params_LMN23_EA.json'
    # param_file = r'Sequim/params_Sequim.json'
    # param_file = r'McCloud_2023_study/params_McCloud2023.json'
    # param_file = r'fishHeart/params_fishHeart.json'
    # param_file = r'McCloud_2024_study/Week_1/params_McCloud.json' 
    run(param_file)