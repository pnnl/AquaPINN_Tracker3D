# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:44:37 2024

@author: chen096
"""
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from scipy import signal
import os
from scipy.io import loadmat
import pandas as pd
import re
from math import sin,cos
import json
import datetime
import itertools
import torch
import logging
from collections import defaultdict
logger = logging.getLogger(__name__)
from ..SharedLibs.decorators import timing
from ..SharedLibs.functions import  replaceFolder
import matplotlib.pyplot as plt

datetime_NegInf = pd.to_datetime("1699-1-1")
datetime_PosInf = pd.to_datetime("2222-12-31")

from scipy.stats import mode
@timing
def modeFilter(dt, width, int_bin):
    # Quantize dt
    dt_int = np.around(dt / int_bin) * int_bin
    n = len(dt_int)
    modes = np.empty(n, dtype=dt_int.dtype)
    counts = np.empty(n, dtype=int)
    
    #half-window size
    half = (width + 1) // 2

    # Initialize frequency dictionary for the first window
    freq = defaultdict(int)
    start = 0
    end = min(n, half)
    for i in range(start, end):
        freq[dt_int[i]] += 1
    
    # Helper function to extract the mode from the frequency dictionary
    def get_mode(freq):
        mode_val, mode_count = None, 0
        for key, count in freq.items():
            if count > mode_count:
                mode_val, mode_count = key, count
        return mode_val, mode_count
    
    for i in range(n):
        # Determine the current window indices
        win_start = max(0, i - half)
        win_end   = min(n, i + half + 1)
        
        # update the frequency counts:
        if i > 0:
            # Remove the element that is no longer in the window
            if i - half - 1 >= 0:
                leaving = dt_int[i - half - 1]
                freq[leaving] -= 1
                if freq[leaving] == 0:
                    del freq[leaving]
            # Add the new element entering the window
            if i + half < n:
                entering = dt_int[i + half]
                freq[entering] += 1
        
        # Compute mode for the current window
        mode_val, mode_count = get_mode(freq)
        modes[i] = mode_val
        counts[i] = mode_count
    return modes, counts

def preSyncPlot(records,outputdir):
    for key in records:   
        phoneID1,phoneID2=key
        toa, dtop,ids = records[key][:,0],records[key][:,1],records[key][:,2]
        pointNum = toa.shape[0]
        if pointNum<5: continue   
        plt.figure()
        t_num=toa-719529
        timestamp=pd.to_datetime(t_num, unit='D').round('s')        
        scatter = plt.scatter(timestamp,dtop, c=ids, cmap=plt.get_cmap('tab10'), s=2)
        plt.legend(*scatter.legend_elements(), loc='best')

        filename = f"Phone{phoneID1}-{phoneID2}.jpg"
        file = os.path.join(outputdir, filename)  
        plt.title(f"Time shift for Phone{phoneID1}-{phoneID2} Point Number {pointNum}")
        plt.tick_params(axis='x', labelrotation=30)
        plt.ylabel("time shift")                  
        plt.savefig(file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

def preSync(loc_sys, decodesFile, badPhones, soundSpeed_ref=1500,
            preSyn_file="", time_start=None, time_end=None, tol_signal=10, width=41):
    time_start = time_start or datetime_NegInf
    time_end = time_end or datetime_PosInf
    
    msg = loc_sys.loadDecodes(decodesFile, converTimeZone=True)    
    phoneInfo, tagInfo=loc_sys.phoneInfo, loc_sys.tagInfo    
    phoneLoc = phoneInfo['XYZ']
    phoneIDs = list(phoneInfo['IDs'])
    phoneIDs_all = list(phoneInfo['IDs_all'])
    records ={}
    records2={}
    for BeaconInd, (BeaconCode,BeaconID) in  enumerate(zip(phoneInfo['AttachedBeacons'],phoneInfo['IDs_all'])):
        if BeaconID in badPhones:
            continue
        if BeaconCode is None:
            continue
        tmp = msg[BeaconInd]['decodes']        
        if tmp is None:
            continue
        decodes=tmp[np.argsort(tmp[:,0]),:]
        t_num=decodes[:,0]-719529
        t_datetime=pd.to_datetime(t_num, unit='D').round('s') 
        ind = (t_datetime >= time_start) & (t_datetime <= time_end)  
        decodes = decodes[ind] 
        for badPhoneID in badPhones:
            decodes=decodes[decodes[:,3].astype(int)!=badPhoneID]
        phoneInd_all  = [phoneIDs.index(i)  for i in decodes[:,3].astype(int)] 
        toa_all = decodes[:, 0]
        dt_all      = np.linalg.norm(phoneLoc[phoneInd_all]-phoneLoc[phoneIDs.index(BeaconID)][None,:], axis=1)/soundSpeed_ref
        top_all = toa_all-dt_all/24/3600  
        if decodes.shape[0]<5:
            continue

        phones  = decodes[:,3].astype(int) 
        PRIs=[]
        for p in np.unique(phones): 
            if np.sum(phones==p)<5:
                continue            
            PRI = np.percentile(np.diff(decodes[:,0][phones==p])*24*3600, 30)
            PRIs.append(PRI)       
        PRI=np.median(np.array(PRIs))
        logger.info(f"PRI for beacon {BeaconID} is {PRI:.2f}")
        
        msg[BeaconInd]['decodes'] = decodes
        nPhone = len(phoneInfo['IDs'])                      
        for i1 in range(nPhone):
            for i2 in range(i1+1, nPhone):
                phoneID1, phoneID2 = phoneIDs[i1],phoneIDs[i2]
                cond = (phones==phoneID1) | (phones==phoneID2)
                if np.sum(cond)<5: continue                 
                ids = phones[cond]          
                toa = toa_all[cond]
                top = top_all[cond]
                dtop=np.diff(top)*np.sign(np.diff(ids))*24*3600   
                dtop[np.diff(ids)==0]=1E10
                ind =np.abs(dtop)<tol_signal
                ind_in_beacon = np.arange(len(top_all))[cond]
                ind_beacon = np.ones_like(top)*BeaconInd
                
                tmp =np.stack((ind_beacon[1:],ind_in_beacon[1:],ind_in_beacon[:-1],
                               toa[1:], dtop), axis=1)[ind]
                ind2 = np.abs(dtop)<15
                
                id_beacon = np.ones_like(top)*BeaconID
                tmp2 = np.stack((toa[1:], dtop, id_beacon[1:]), axis=1)[ind2]
                
                key = (phoneID1, phoneID2)
                if key not in records:
                    records[key] = tmp
                    records2[key] = tmp2
                else:
                    records[key]  = np.concatenate((records[key], tmp), axis=0)               
                    records2[key] = np.concatenate((records2[key], tmp2), axis=0)               
                    
    preSyncPlot(records2, outputdir=os.path.join(os.path.dirname(preSyn_file),"sync", "preSync"))
                    
    # get mode
    decodes_all=np.zeros((0,4))
    beaconInd_all=np.zeros((0))
    ind_in_beacon_all=[np.zeros(0, dtype='int')]*len(msg)
    logger.info("mode filter...")
    for key in records: 
        phoneID1,phoneID2=key
        logger.info(f"pair {phoneID1}-{phoneID2}")
        ind_beacon, ind_in_beacon1,ind_in_beacon2 = records[key][:,0].astype(int), records[key][:,1].astype(int),records[key][:,2].astype(int)              
        toa, dtop=records[key][:,-2], records[key][:,-1]
        ind_sort = np.argsort(toa)
        dtop = dtop[ind_sort]
        max_filtered_points = None
        results=[]
        
        for int_bin in [5E-3, 2E-3, 1E-3, 8E-4, 5E-4]:
            modes, nums = modeFilter(dtop, width,int_bin)  
            ind = (np.abs(modes-dtop)<=int_bin/2 ) & (nums>=3) 
            if max_filtered_points is None:
                max_filtered_points = np.sum(ind)
            else:
                max_filtered_points = max(max_filtered_points, np.sum(ind))
            print(int_bin, np.sum(ind), max_filtered_points, np.sum(ind)/max_filtered_points)
            if np.sum(ind)<max_filtered_points*0.9:                
                break 
            else:
                results.append([modes,nums, ind, int_bin])
        modes,nums, ind, int_bin = results[-1]        
        ind_beacon   =ind_beacon[ind_sort][ind]
        ind_in_beacon1=ind_in_beacon1[ind_sort][ind]
        ind_in_beacon2=ind_in_beacon2[ind_sort][ind]     
        
        for BeaconInd in np.unique(ind_beacon):
            ind = ind_beacon==BeaconInd
            ind_in_beacon = np.unique(np.concatenate((ind_in_beacon1[ind], ind_in_beacon2[ind])))
            # thisDecodes = msg[BeaconInd]['decodes']
            ind_in_beacon_all[BeaconInd] = np.concatenate((ind_in_beacon_all[BeaconInd], ind_in_beacon))
            # decodes_pair=thisDecodes[ind_in_beacon]
            # decodes_all = np.concatenate( (decodes_all, decodes_pair,) )
            # beaconInd_all = np.concatenate( (beaconInd_all, BeaconInd*np.ones(decodes_pair.shape[0])) )
    #  resort decodes from decodes_all
    new_msg=[{"decodes":None} for _ in msg]
    for BeaconInd in range(len(msg)):
        if msg[BeaconInd]['decodes'] is None: continue
        new_msg[BeaconInd]['decodes'] = msg[BeaconInd]['decodes'][np.unique(ind_in_beacon_all[BeaconInd])]
    
    for BeaconInd in range(len(msg)):
        n_old =    msg[BeaconInd]['decodes'].shape[0] if     msg[BeaconInd]['decodes'] is not None else 0
        n_new =new_msg[BeaconInd]['decodes'].shape[0] if new_msg[BeaconInd]['decodes'] is not None else 0
        logger.info(f"keep ratio for beacon {phoneIDs_all[BeaconInd]}: {n_new}/{n_old}={n_new/(n_old+1E-10)*100:.2f}%")
    with open(preSyn_file, 'wb') as file:
        pickle.dump(new_msg, file)    
    # for key in records:   
    #     phoneID1,phoneID2=key
    #     toa, dtop = records[key][:,0], records[key][:,-1]*24*3600
    #     t_num=toa-719529
    #     t_datetime=pd.to_datetime(t_num, unit='D').round('s') 
    #     plt.plot(t_datetime, signal.medfilt(dtop, 1),'.k', markersize=1)              
    #     # ymax = np.percentile(dtop, 97)
    #     plt.ylim(ymax=120)
    #     filename = f"Phone{phoneID1}-{phoneID2}.jpg"
    #     file = os.path.join(outputdir, filename)  
    #     plt.title(f"Time shift for Phone{phoneID1}-{phoneID2} Point Number {toa.shape[0]}")
    #     plt.tick_params(axis='x', labelrotation=30)
    #     plt.ylabel("time shift")                  
    #     plt.savefig(file, dpi=300, bbox_inches='tight', facecolor='white')
    #     # plt.close()  
    # file = os.path.join(outputdir, 'records.pickle')
    # with open(file, 'wb') as f:
    #     pickle.dump(records, f)            
    return   






class TOAinfo_creator():    
    def __init__(self, loc_sys, decodesFile, tol_signal=0.3,  update=True,
                 badPhones=[], time_start=None, time_end=None,
                 syncFile=None, filesuffix=None,soundSpeed_ref=1500,
                 tagIndices=None, newFolder=None):
        self.timeZone_local =loc_sys.timeZone_local
        '''
        creat TOA from a .mat file
        Parameters
        ----------
        decodesFile : str
            .mat file name.
        tol : float, optional
            two toa from two receivers are deemed as a signal when their difference is smaller than $tol$. 
            The default is 0.3.
        update : bool, optional
            whether to update the created TOA file. The default is False.
        '''
        
        filesuffix = filesuffix or ""
        preSync_file = os.path.splitext(decodesFile)[0] + filesuffix + '_preSync.pickle' 
        TOA_file = os.path.splitext(decodesFile)[0] + filesuffix + '_preSync_TOA.pickle' 
        if newFolder is not None:
            preSync_file=replaceFolder(preSync_file, newFolder) 
            TOA_file=replaceFolder(TOA_file, newFolder) 
            
        
        preSync(loc_sys, decodesFile, badPhones, soundSpeed_ref=soundSpeed_ref,
                    preSyn_file=preSync_file, time_start=time_start, time_end=time_end,
                    tol_signal=tol_signal, width=100)       
        
        # generate,update or keep the TOA file
        if update or (not os.path.isfile(TOA_file)):
            msg = loc_sys.loadDecodes(preSync_file, converTimeZone=False)             
            TOAinfo = []
            for curTag in msg:
                TOAinfo_iTag=self.creat_one_tag(curTag,tol_signal, 
                                                time_start=time_start, 
                                                time_end=time_end,
                                                syncFile=syncFile)
                self.apply_TOAD_filter(TOAinfo_iTag)
                TOAinfo.append(TOAinfo_iTag)
            with open(TOA_file, 'wb') as file:
                pickle.dump(TOAinfo, file)
        else: 
            with open(TOA_file, 'rb') as file:
                TOAinfo=pickle.load(file)
        
        self.TOAinfo = TOAinfo
        self.nTag = len(self.TOAinfo)
        phoneIDs=set()
        for TOAinfo_iTag in self.TOAinfo:
            if TOAinfo_iTag is not None:
                phoneIDs=phoneIDs.union(set(TOAinfo_iTag['phoneIDs']))
        self.phoneIDs=sorted(list(phoneIDs))
        self.nPhone = len(phoneIDs)
        self.df, self.df2=self.check()
        
        
    def getTOAinfo(self, TagID, time_start=None, time_end=None):
        if isinstance(TagID, int):
            curTOA = self.TOAinfo[TagID]
            return self.trimTOA(curTOA, time_start, time_end)
        elif isinstance(TagID, (list, tuple)):
            res=[]
            for iTagID in TagID:
                curTOA = self.TOAinfo[TagID]
                res.append(self.trimTOA(curTOA, time_start, time_end))
            return res
        else:
            raise Exception(f"unknown TagID:{TagID}")
        return 
    
    def trimTOA(self, curTOA, time_start=None, time_end=None):
        if curTOA is None:
            return None
        time_start = time_start or datetime_NegInf
        time_end = time_end or datetime_PosInf
        t_datetime = curTOA['TOA_t_datetime']
        ind = (time_start<=t_datetime) & (t_datetime<=time_end)
        TOAinfo = {'TOA':curTOA['TOA'][ind],
                   'TOA_t_num':curTOA['TOA_t_num'][ind],
                   'TOA_t_datetime':curTOA['TOA_t_datetime'][ind],
                   'SNR':curTOA['SNR'][ind],
                   'phoneIDs':curTOA['phoneIDs'],
                   }
        return TOAinfo
    def check(self):
        "print a sumary of TOA info"
        nPhone, nTag = self.nPhone, self.nTag
        stat = np.zeros((nPhone, nTag)).astype(int)
        stat2= np.zeros((nPhone, nTag)).astype(int)
        for i in range(self.nTag):
            if self.TOAinfo[i] is None:
                continue
            nvalid = np.sum(self.TOAinfo[i]['TOA']>0, axis=1)            
            for j in range(nPhone):
                stat[j,i] = np.sum( (nvalid>=j+1) )
                phoneIDs=list(self.TOAinfo[i]['phoneIDs'])
                if self.phoneIDs[j] in phoneIDs:
                    ind_j = phoneIDs.index(self.phoneIDs[j])
                    stat2[j,i]= np.sum(self.TOAinfo[i]['TOA'][:,ind_j]>0)
        df = pd.DataFrame(data=stat, 
                          index=np.arange(nPhone)+1, 
                          columns=np.arange(nTag))
        df2 = pd.DataFrame(data=stat2, 
                          index=self.phoneIDs, 
                          columns=np.arange(nTag))        
        
        logger.info(f'df: \n {df}')
        logger.info(f'df2: \n {df2}')
        return df,df2
    @timing
    def creat_one_tag(self,curTag, tol_signal=0.3,time_start=None, time_end=None,syncFile=None):     
        "build TOA info for a tag"
        info_mat = curTag['decodes'] # date, message time, SNR, phone_ID
        TOAinfo={'TOA':np.empty((0,0)),
                 'TOA_t_num':np.empty(0),
                 'TOA_t_datetime':pd.to_datetime(np.empty(0), unit='D').round('s'),
                 'phoneIDs':np.empty(0),
                 'SNR':np.empty(0),
                 }
        if info_mat is None or len(info_mat)==0:
            return TOAinfo
        t=  info_mat[:,1]*24*3600
        phoneIDs, ind_phones = np.unique(info_mat[:,3].astype(int),return_inverse=True)
        t=self.sync(t,phoneIDs, ind_phones, syncFile)
        nt = len(t)
        ind = np.argsort(t)
        info_mat = info_mat[ind,:]
        ind_phones = ind_phones[ind]
        phoneIDs, ind_phones = np.unique(info_mat[:,3].astype(int),return_inverse=True)
        t        = t[ind]        
        #phoneIDs, ind_phones = np.unique(info_mat[:,3].astype(int),return_inverse=True)
        #t=self.sync(t,phoneIDs, ind_phones, syncFile)
        dt = np.diff(t)        
        # build CSR matrix
        indptr   = np.where(dt>tol_signal)[0] + 1
        indptr   = np.hstack( (0, indptr, nt))        
        indices  = ind_phones
        shape=(len(indptr)-1, indices.max()+1)       
        #remove duplicate data
        ind_uni, indices, indptr = self.remove_duplicate_toa(np.arange(len(t)), indices, indptr, shape, t, tol_signal)
        TOAinfo['TOA']      = csr_matrix((t[ind_uni], indices, indptr), shape=shape).toarray()
        TOAinfo['TOA_t_num'] = info_mat[ind_uni][indptr[:-1],0]-719529
        TOAinfo['TOA_t_datetime'] = pd.to_datetime(TOAinfo['TOA_t_num'], unit='D').round('s')
        SNR = info_mat[:,2]
        TOAinfo['SNR']      = csr_matrix((SNR[ind_uni], indices, indptr), shape=shape).toarray()
        TOAinfo['phoneIDs'] = phoneIDs
        #TOAinfo['min_days'] = np.floor(np.min(days))        
        for key in curTag.keys():
            if 'decodes' not in key and not key.startswith('__'):
                TOAinfo[key]=curTag[key]
        TOAinfo = self.trimTOA(TOAinfo, time_start, time_end)        
        return TOAinfo   

    def remove_duplicate_toa(self, data, indices, indptr, shape, t, tol):
       new_data = []
       new_indices = []
       new_indptr = [0]
       n_rows = shape[0]
       for row in range(n_rows):
           # Get the start and end position of the current row
           start, end = indptr[row], indptr[row + 1]
           row_indices = indices[start:end]
           row_data = data[start:end]
           t_data   = t[start:end]
           # Use a dictionary to track seen column indices (keep the first occurrence)
           seen = {}
           seen_t = {}
           for idx, col in enumerate(row_indices):
               if col not in seen:
                   seen[col] = row_data[idx]
                   seen_t[col]=t_data[idx]
           # remove exception
           t_min = np.array(list(seen_t.values())).min()
           seen = {col:seen[col] for col in seen if seen_t[col]<t_min+tol}
           
           # Sort the results (to maintain sorted column indices)
           sorted_indices = sorted(seen.keys())           
           sorted_data = [seen[col] for col in sorted_indices]
           # Update the result
           new_indices.extend(sorted_indices)
           new_data.extend(sorted_data)
           new_indptr.append(len(new_indices))
       return new_data, new_indices, new_indptr  
    
    def sync(self, t,phoneIDs, ind_phones, syncFile):
        if syncFile is None:
            return t
        with open(syncFile, "rb") as f: 
            timeSync=pickle.load(f)
        coeff_sync = timeSync['timeSync']  
        tmin_sync,tmax_sync=timeSync['tmin'], timeSync['tmax']
        for i,ID in enumerate(phoneIDs):
            a,b,c=coeff_sync[ID]
            ind=ind_phones==i
            tnorm=(t[ind]-(tmax_sync+tmin_sync)/2)/(tmin_sync-tmax_sync)*2
            logger.debug(f'{tnorm.min(), tnorm.max()}')
            t[ind]+=a*tnorm**2 + b*tnorm + c        
        return t

    @classmethod
    def split_one_tag(cls, TOAinfo, tol_split=np.inf, testInfo=''):
        All_t_datetime = TOAinfo['TOA_t_datetime']
        test_periods = [[All_t_datetime.min(), All_t_datetime.max()]]
        testNames    = ['All']
        if testInfo is not None:
            test_periods, testNames= testInfo['periods'], testInfo['names']
        intervals = []
        names=[]
        for (time_test_start, time_test_end), testName in zip(test_periods,testNames):             
            ind = (time_test_start<=All_t_datetime) & (All_t_datetime<=time_test_end)
            if np.sum(ind)<10:
                continue
            TOA_t_num=TOAinfo['TOA_t_num'][ind]
            TOA_t_datetime = TOAinfo['TOA_t_datetime'][ind]
            dt = np.diff(TOA_t_num)*24*3600
            inds=np.where(dt>tol_split)[0]
            datetimesL = [TOA_t_datetime[0]] + list(TOA_t_datetime[inds+1]) 
            datetimesR = list(TOA_t_datetime[inds])  + [TOA_t_datetime[-1]]
            
            i=0
            for time_s, time_e in list(zip(datetimesL, datetimesR)):
                nSignal = np.sum( (time_s<=TOA_t_datetime) & (TOA_t_datetime<=time_e))
                if nSignal<10:
                    pass
                else:
                    intervals.append( (time_s, time_e) )
                    names.append(testName+f"_interval{i}")
                    i+=1
        return intervals,names
    @timing 
    def apply_TOAD_filter(self, TOAinfo_Tagi, nstd=3,window_size=11):
        if TOAinfo_Tagi is None:
            return None
        TOA=TOAinfo_Tagi['TOA']
        if len(TOA)==0:
            return TOAinfo_Tagi
        mask=TOA>0        
        TOA[~mask] = np.nan
        nPhone = TOA.shape[1]
        for k in range(10):
            indictor=np.zeros_like(TOA).astype(int)
            for i in range(nPhone):
                for j in range(i+1,nPhone):     
                    TDOA = TOA[:,i]-TOA[:,j]
                    if np.all(np.isnan(TDOA)):
                        continue
                    ind_all = np.where(np.isfinite(TDOA))[0]
                    if len(ind_all)<window_size:
                        continue
                    TDOA_tmp = TDOA[np.isfinite(TDOA)]
                    TDOA_filter=signal.medfilt(TDOA_tmp, window_size)	
                    err = TDOA_tmp-TDOA_filter		
                    size = window_size                    
                    window = np.ones(size)/size
                    mu = np.convolve(err, window,'same')
                    var= np.convolve(err**2, window, 'same')                    
                    std  = np.sqrt(var-mu**2)
                    cond = np.abs(err-mu)>nstd*std
                    sw   = (window_size-1)//2
                    #cond[:sw] = np.abs(TDOA_tmp[:sw]-np.mean(TDOA_tmp[:2*sw]))>nstd*np.std(TDOA_tmp[:2*sw])
                    #cond[-sw:] = np.abs(TDOA_tmp[-sw:]-np.mean(TDOA_tmp[-2*sw:]))>nstd*np.std(TDOA_tmp[-2*sw:])
                    cond2 = np.abs(err-err.mean())>nstd*err.std()
                    ind=np.where(cond | cond2)[0]
                    ind=ind_all[ind]
                    indictor[ind,i] += 1;  indictor[ind,j] += 1
            badPoints = (indictor==np.max(indictor, axis=1)[:,None]) & (indictor>0)
            TOA[badPoints] = np.nan
            logger.info(f"iter:{k}: {np.sum(badPoints)} bad points removed")
            if np.sum(badPoints)==0:
                break
        TOA[np.isnan(TOA)] = -999
        ind_keep=np.any(TOA>0, axis=1)
        TOAinfo_Tagi['TOA'] = TOA[ind_keep]
        TOAinfo_Tagi['TOA_t_num'] = TOAinfo_Tagi['TOA_t_num'][ind_keep]
        TOAinfo_Tagi['TOA_t_datetime'] = TOAinfo_Tagi['TOA_t_datetime'][ind_keep]
        TOAinfo_Tagi['SNR'] = TOAinfo_Tagi['SNR'][ind_keep]        
        return TOAinfo_Tagi

   
                  
    

