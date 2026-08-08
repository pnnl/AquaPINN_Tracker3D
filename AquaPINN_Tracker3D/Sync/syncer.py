# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:21:41 2024

@author: chen096
"""

import os
#import torch,sys
import time
from ..file import fileSystem
from .ML import Model, getTrainData
from .dataLoader import TOAinfo_creator, datetime_NegInf, datetime_PosInf,preSync
from ..SharedLibs.functions import load_params, replaceFolder, soundSpeed_freshWater
from ..SharedLibs.visualization import  showDetections
import json
import logging
import pandas as pd
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from ..SharedLibs.decorators import timing
plt.close('all')

@timing
def run(param_file, loc_sys):
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('+++++++++++++++++++ SYNC ... ++++++++++++++++++++++')
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    phoneInfo, tagInfo=loc_sys.phoneInfo, loc_sys.tagInfo    
    # load parameters
    train_params = load_params(param_file)[0]
    outFolder = os.path.join(
        train_params["output_folder"], 'sync')
    FNS = fileSystem(outFolder,)    
    lr_lambda = lambda step:  0.98**(step//250)
    badPhones = train_params['badPhones']
    TOAinfo_update=True
    outputFolder=train_params['output_folder']
    fartherPath=outputFolder
    tagFile = train_params['tagFile']
    decodesFile = loc_sys.getDecodesFile(tagFile, 'Beacon', newFolder=fartherPath)   
    syncFile  = os.path.splitext(decodesFile)[0] + '_syncInfo.pickle'  
    syncFile  = replaceFolder(syncFile, outputFolder)
    tagSyncedFile=loc_sys.getSyncedDecodesFile(tagFile, 'Tag', newFolder=outputFolder)
    if False and os.path.isfile(syncFile) and os.path.isfile(tagSyncedFile):
        logger.info(f"sync file already exists: {syncFile} and {tagSyncedFile}")
        return
    time_start = pd.to_datetime(train_params.get('time_start', datetime_NegInf)
                                ) 
    time_end = pd.to_datetime(train_params.get('time_end', datetime_PosInf)
                              ) 
    #if time_end - time_start<pd.Timedelta(hours=3):
    # time_start += pd.Timedelta(hours=-1)
    # time_end   += pd.Timedelta(hours=1)  
    TOAinfo = TOAinfo_creator(loc_sys,decodesFile=decodesFile,
                              tol_signal=train_params["tol_signal_beacon_preSync"],
                              update=TOAinfo_update, badPhones=badPhones,
                              time_start=time_start+pd.Timedelta(minutes=-30),
                              time_end=time_end+pd.Timedelta(minutes=30),
                              soundSpeed_ref =soundSpeed_freshWater(train_params["temperature_ref"]),
                              newFolder=outputFolder)
    TOA_all=dict()
    for BeaconInd, (BeaconCode,BeaconID) in  enumerate(zip(phoneInfo['AttachedBeacons'],phoneInfo['IDs_all'])):
        if BeaconID in train_params['badPhones']:
            continue
        if BeaconCode is None:
            continue
        TOA_all[BeaconID] = TOAinfo.getTOAinfo(TagID=BeaconInd)
    
    detectionOutputFolder=os.path.join(outFolder, 'detections')    
    # showDetections(decodesFile, tagType='Beacon', 
    #                outputdir=detectionOutputFolder, time_start=time_start,
    #                time_end=time_end,timeZone=loc_sys.timeZone_local)        
    ignoreZ =train_params["ignoreZ"].lower() == 'true'
    # Start timing
    start_total_time = time.time()
    # get data
    data = getTrainData(TOA_all,loc_sys=loc_sys,
                        ignorePhoneNum=train_params["ignorePhoneNum"],
                        ignoreZ=ignoreZ)
    if data is None:
        logger.error('data is None')
        return        
    # run
    model = Model(FNS, data,  train_params["dam"],err_GPS=train_params["err_GPS"],
                  temperature_ref=train_params["temperature_ref"], dis_detect=train_params["dis_detect"],
                  region_lb=train_params["region_lb"], region_ub=train_params["region_ub"],                      
                  ignoreZ=ignoreZ,
                  toSolveSync=True, toSolveLoc=True)
    # train model
    train=True
    if train:
        problematicPhones=[]
        for i in range(2): 
            logger.info(f" train iter {i} ...")
            nIterAdam_sync = train_params.get("nIterAdam_sync",
                                              train_params.get("nIterAdam", 10000))
            nIterBFGS_sync = train_params.get("nIterBFGS_sync",
                                              train_params.get("nIterBFGS", 100))
            model.train(nIterAdam=nIterAdam_sync, nIterBFGS=nIterBFGS_sync,
                        lr=0.001, lr_lambda=lr_lambda, 
                        problematicPhones=problematicPhones              )
            problematicPhones=model.summary(syncFile)
            if len(problematicPhones)==0:
                break
          
    # End timing
    model.loadParas() #status='best')
    model.summary(syncFile)  
    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    string = "Total time: {:.4f} sec".format(total_time)
    logger.info(string)
    with open(FNS.params(), 'w') as f:
        json.dump(train_params, f, sort_keys=False, indent=4)
    
    model.plot(syncFile=syncFile)               
    model.showGraph(loc_sys, syncFile=syncFile)
    
    decodesFile = loc_sys.getDecodesFile(tagFile, 'Tag', newFolder=fartherPath) 
    loc_sys.applySync2Decodes(decodesFile=decodesFile,
                             syncFile=syncFile,
                             badPhones=train_params['badPhones'],
                             newFolder=outputFolder,
                             time_start=time_start,
                             time_end=time_end)
    decodesFile = loc_sys.getDecodesFile(tagFile, 'Beacon', newFolder=fartherPath)
    loc_sys.applySync2Decodes(decodesFile=decodesFile,
                             syncFile=syncFile,
                             badPhones=train_params['badPhones'],
                             newFolder=outputFolder,
                             time_start=time_start,
                             time_end=time_end)
        
    
    loc_sys.applySync2Phone(phoneFile=train_params["phoneFile"], newFolder=outputFolder,
                            syncFile=syncFile)
    
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('+++++++++++++++++++ SYNC done ++++++++++++++++++++++')
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')    
    return 
    

if __name__=='__main__':
    # shad tag EA
    # param_file   = r'Shad_EA/params_shad.json'
    # param_file   = r'Shad_beacons/params_shad_beacons.json'
    # param_file = r'LMN23_dam_beacons/params_LMN23_EA.json'
    param_file = r'Levy_Park/params_levy_park.json'
    # param_file = r'LMN23_EA/params_LMN23_EA.json'
    # param_file = r'Sequim/params_Sequim.json'
    # param_file = r'McCloud_2023_study/params_McCloud2023.json'
    param_file = r'fishHeart/params_fishHeart.json'
    param_file = r'McCloud_2024_study/Week_1/params_McCloud.json' 
    run(param_file)
