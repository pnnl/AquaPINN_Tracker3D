# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:21:41 2024

@author: chen096
"""
import os
import numpy as np
#import torch,sys
import time
from ..file import fileSystem
from . import ML_NN
from .AML import AMLSolver
from ..SharedLibs.functions import load_params
from .dataLoader import TOAinfo_creator
from .postProcess import plot_track_OneFigureEqual, addGPS2Track
import json
import logging
import pickle
from scipy.io import loadmat
logger = logging.getLogger(__name__)
from ..SharedLibs.decorators import timing

import matplotlib.pyplot as plt
plt.close('all')


lr_lambda = lambda step:  0.99**(step//500)


@timing
def run(param_file, loc_sys, trackBeacon=False):
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('+++++++++++++++++++ TRACK ... +++++++++++++++++++++')
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')    
    phoneInfo, tagInfo ,testInfo=loc_sys.phoneInfo, loc_sys.tagInfo, loc_sys.testInfo
    testInfo = testInfo if not trackBeacon  else None
    # load parameters
    train_params_list = load_params(param_file)
    #
    tagFile = train_params_list[0]['tagFile']
    decodesFile = loc_sys.getSyncedDecodesFile(tagFile, 'Tag' if not trackBeacon else 'Beacon',
                                               newFolder=train_params_list[0]['output_folder']) 
    nTag = 1
    if train_params_list[0]["TagID"] == -1:
        with open(decodesFile, 'rb') as f:
            data = pickle.load(f)
            nTag = len(data)
    # start model training process    
    kk=-1
    for index_tag, index_params in [(iii,jjj) for iii in  range(nTag) for jjj in range(len(train_params_list))] :
        kk += 1
        track_Result_all=dict()
        train_params = train_params_list[index_params]
        if nTag>1:
            train_params["TagID"]=index_tag
        badPhones = train_params['badPhones']  
        TOAinfo_update=True if kk==0 else False
        if not trackBeacon:
            TagCode = tagInfo['code'][index_tag if nTag>1 else train_params["TagID"]]
            isBeacon = TagCode in phoneInfo['AttachedBeacons']
        else:
            isBeacon = True
            TagCode = phoneInfo['AttachedBeacons'][index_tag if nTag>1 else train_params["TagID"]]
        beaconID = None if not isBeacon else phoneInfo['AttachedBeacons'].index(TagCode)        
                
        trackMethod = train_params.get('trackMethod', 'AML')
        assert trackMethod in ['AML', 'NN']         
        TOAinfo = TOAinfo_creator(loc_sys,decodesFile=decodesFile,
                                  tol_signal=train_params["tol_signal"],
                                  update=TOAinfo_update, badPhones=badPhones,
                                  time_start=train_params.get('time_start', None),
                                  time_end=train_params.get('time_end', None),
                                  filesuffix=None,)
        ignoreZ =train_params["ignoreZ"].lower() == 'true'
        split_by_TestFile = train_params['split_by_TestFile'].lower() == 'true'
        tol_split = train_params["tol_split"] if train_params['split_by_tol'].lower() == 'true' else np.inf
        intervals, names = TOAinfo.split_one_tag(TOAinfo.getTOAinfo(TagID=train_params["TagID"]),
                                                 tol_split=tol_split, testInfo=testInfo if split_by_TestFile else None,
                                                 numTrack=8000 if trackMethod not in ['AML'] else int(1E6),
                                                  dim=2 if ignoreZ else 3)   
        for ind_interval, ((time_interval_start, time_interval_end),name_interval) in enumerate(zip(intervals,names)):
            logger.info(f"<<<<<<<<<<<<<<<<< {name_interval} >>>>>>>>>>>>>>>")
            logger.info(f"{ind_interval}/{len(names)}")
            interval_num = len(names)
            is_last_interval = True if ind_interval==len(names)-1 else False
            TOAinfo_slice = TOAinfo.getTOAinfo(TagID=train_params["TagID"],
                                             time_start=time_interval_start,
                                             time_end=time_interval_end)
    
            timestamp = time_interval_start.strftime('%Y-%m-%d-%H-%M')+'_'+time_interval_end.strftime('%Y-%m-%d-%H-%M')
            TagStr = "Tag"+str(train_params["TagID"])+"_"+TagCode
            outFolder = os.path.join(
                train_params["output_folder"],
                'Track' if not trackBeacon else 'TrackBeacon')         
            layers = [train_params["first_layer_nodes"]]+[train_params["hidden_layer_nodes"]
                                                          ]*train_params["hidden_layers"]+[train_params["last_layer_nodes"]]
            # Start timing
            start_total_time = time.time()
            # get data
            data = ML_NN.getTrainData(TOAinfo_slice,
                                phoneInfo=phoneInfo, ignorePhoneNum=train_params["ignorePhoneNum"],
                                ignoreZ=ignoreZ,dam=train_params["dam"])
            if data is None:
                logger.warning('data is None')
                continue
            track_GPS = None
            if os.path.isfile(train_params["GPSFile"]):
                try:
                    track_GPS = loc_sys.loadGPS(train_params["GPSFile"])
                except Exception as e:
                    logger.warning(f"Failed to load GPS file {train_params['GPSFile']}: {e}")
            if isBeacon:
                dt = 1
                import pandas as pd
                ts=pd.Timestamp(time_interval_start)
                te=pd.Timestamp(time_interval_end)
                t_num =np.arange(ts.timestamp(),te.timestamp(),dt)
                t_datetime = np.array([ ts + pd.Timedelta(seconds=t-ts.timestamp()) for t in t_num])
                ENE0, theta=phoneInfo['ENE0'], phoneInfo['theta']        
                XYZ_tmp= loc_sys.XYZ2ENE(phoneInfo['XYZ_all'], ENE0, theta)
                locBeacon = XYZ_tmp[beaconID][None,:]
                track_GPS = {'XYZ': np.tile(locBeacon,(len(t_num),1)),
                             't_datetime':t_datetime,
                             't_num':t_num,
                             'GPS_Z_valid':True}                             
                
            feature_num = train_params["first_layer_nodes"]
            mask = data[-2]
            nvalid = np.sum(mask, axis=1)
            period_factor = train_params.get("period_factor", 300)
            Vel_ref = np.sum(nvalid >= 4)/period_factor
            #if ignoreZ:
            #    Vel_ref = min( max(np.sum(nvalid >= 3)/period_factor,0.5), feature_num/5)
            #else:
            #    Vel_ref = min(max(np.sum(nvalid >= 4)/period_factor,0.5), feature_num/5)
            logger.info(f"Vel_ref={Vel_ref}, npoints={np.sum(nvalid >= 4)}/{len(nvalid)}")
            data_fed = data  
            assert trackMethod in ['AML', 'NN']
            nSamples = train_params.get('nSamples', 1)
            trackMethod_display='NNplus' if trackMethod=='NN' and nSamples>1 else trackMethod
            Model = ML_NN.Model   
            FNS = fileSystem(outFolder,'' , '_'+TagStr)
            saveCSV=FNS.track(trackMethod)
            if trackMethod=='AML':                
                track_Result=AMLSolver(data, region_lb=train_params["region_lb"], region_ub=train_params["region_ub"],
                                        ignoreZ=ignoreZ,
                                        temperature_ref=train_params["temperature_ref"], 
                                        dis_detect=train_params["dis_detect"],
                                        RSR=train_params.get('sdfNetFile', None))
                if track_Result is None:
                    logger.warning('no AML is obtained')
                    continue
                track_Result = Model.XYZ2ENE(loc_sys, track_Result)
            elif trackMethod == 'NN':
                Model = ML_NN.Model
                model = Model(FNS, data_fed, train_params["TagID"], train_params["dam"],
                              temperature_ref=train_params["temperature_ref"], dis_detect=train_params["dis_detect"],
                              region_lb=train_params["region_lb"], region_ub=train_params["region_ub"],
                              ignorePhoneNum=train_params["ignorePhoneNum"],
                              Vel_ref=Vel_ref, ignoreZ=ignoreZ, 
                              sdfNetFile=train_params.get('sdfNetFile', None),
                              distance_threshold=train_params.get("distance_threshold", 3.0),
                              std_deviation_factor=train_params.get("std_deviation_factor", 5.0),
                              time_shift_factor=train_params.get("time_shift_factor", 0.3),
                              uq_std=train_params.get("uq_std", 1.0),
                              loc_sys=loc_sys,
                              layers=layers)     
                saveCSV  =FNS.track(trackMethod_display)
                track_Result=model.train(nIterAdam=train_params["nIterAdam"], nIterBFGS=train_params["nIterBFGS"],
                            lr=train_params["lr"], lr_lambda=lr_lambda, weight_decay=train_params["weight_decay"],
                            nSamples=nSamples,
                            sampleMethod=train_params.get('sampleMethod', 'no noise'),)                         
            # End timing
            end_total_time = time.time()
            total_time = end_total_time - start_total_time
            string = "Total time for combo {}/{}, interval {}/{}: {:.4f} sec".format(
                index_params, len(train_params_list), ind_interval, len(intervals), total_time)
            logger.info(string)
            with open(FNS.params(), 'w') as f:
                json.dump(train_params, f, sort_keys=False, indent=4)
            
            track_Result_all=Model.trackCat(track_Result_all, track_Result)
            if not is_last_interval: continue
            track_Result=track_Result_all

            vel=np.diff(track_Result['XYZ'], axis=0)/(np.diff(track_Result['t_num'])[:,None]+1E-10)
            vel_mag=np.linalg.norm(vel, axis=1)
            logger.info(f'tagid={train_params["TagID"]}, dt_max={np.diff(track_Result["t_num"]).max()}, vel_max={vel_mag.max()}')
                        
            plot_track_OneFigureEqual(track_Result,  track_GPS,
                                    fout=FNS.figTrack(trackMethod_display), testInfo=testInfo if not split_by_TestFile else None, description=TagStr, method=trackMethod_display.replace('plus',''))   
            Model.writeTrack(addGPS2Track(track_Result,track_GPS), 
                            saveCSV=saveCSV)      
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info('+++++++++++++++++++ TRACK done ++++++++++++++++++++')
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++') 
    return              

