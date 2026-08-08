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
import datetime
import logging
logger = logging.getLogger(__name__)
from ..SharedLibs.decorators import timing

datetime_NegInf = datetime.datetime(1699,1,1)
datetime_PosInf = datetime.datetime(2222,12,31) 
str2datetime = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


class TOAinfo_creator():
    def __init__(self,  loc_sys, decodesFile, tol_signal=0.3,  update=True,
                 badPhones=[], time_start=None, time_end=None,
                 filesuffix=None,phoneInfo=None,):
        self.timeZone_local =loc_sys.timeZone_local
        '''
        creat TOA from a .mat file
        Parameters
        ----------
        decodesFile : str
            .pickle file name.
        tol : float, optional
            two toa from two receivers are deemed as a signal when their difference is smaller than $tol$. 
            The default is 0.3.
        update : bool, optional
            whether to update the created TOA file. The default is False.
        '''
        if filesuffix is None:
            TOA_file = os.path.splitext(decodesFile)[0] + '_TOA.pickle' 
        else:
            TOA_file = os.path.splitext(decodesFile)[0] + filesuffix +'.pickle'
        
        
        
        # generate,update or keep the TOA file
        if update or (not os.path.isfile(TOA_file)):
            msg = loc_sys.loadDecodes(decodesFile) 
            TOAinfo = []
            for curTag in msg:
                TOAinfo_iTag=self.creat_one_tag(curTag,tol_signal, badPhones=badPhones,
                                                time_start=time_start, time_end=time_end,
                                                loc_sys=loc_sys)
                self.apply_TOAD_filter(TOAinfo_iTag)
                TOAinfo.append(TOAinfo_iTag)
            with open(TOA_file, 'wb') as file:
                pickle.dump(TOAinfo, file)
        else: 
            with open(TOA_file, 'rb') as file:
                TOAinfo=pickle.load(file)        
        self.TOAinfo = [self.trimTOA(curTOA, time_start, time_end) for curTOA in TOAinfo]
        self.nTag = len(self.TOAinfo)
        phoneIDs=set()
        for TOAinfo_iTag in self.TOAinfo:
            phoneIDs=phoneIDs.union(set(TOAinfo_iTag['phoneIDs']))
        self.phoneIDs=list(phoneIDs)
        self.nPhone = len(phoneIDs)
        self.df=self.check(TOA_file)
        
    def getTOAinfo(self, TagID, time_start=None, time_end=None):
        curTOA = self.TOAinfo[TagID]
        return self.trimTOA(curTOA, time_start, time_end)
    def trimTOA(self, curTOA, time_start=None, time_end=None):
        time_start = time_start or datetime_NegInf
        time_start = str2datetime(time_start) if isinstance(time_start, str) else time_start
        time_end = time_end or datetime_PosInf
        time_end = str2datetime(time_end) if isinstance(time_end, str) else time_end
        t_datetime = curTOA['TOA_t_datetime']
        ind = (time_start<=t_datetime) & (t_datetime<=time_end)
        TOAinfo = {'TOA':curTOA['TOA'][ind],
                   'TOA_t_num':curTOA['TOA_t_num'][ind],
                   'TOA_t_datetime':curTOA['TOA_t_datetime'][ind],
                   'SNR':curTOA['SNR'][ind],
                   'phoneIDs':curTOA['phoneIDs'],
                   }
        return TOAinfo   
    def check(self, TOA_file):
        "print a sumary of TOA info"
        nPhone, nTag = self.nPhone, self.nTag
        stat  = np.zeros((nPhone, nTag)).astype(int)
        stat2 = np.zeros((nPhone, nTag)).astype(int)
        for i in range(self.nTag):
            nvalid = np.sum(self.TOAinfo[i]['TOA']>0, axis=1)
            phoneIDs_valid=self.TOAinfo[i]['phoneIDs']
            for j in range(len(phoneIDs_valid)):
                jj=self.phoneIDs.index(phoneIDs_valid[j])
                stat[j,i]  = np.sum( (nvalid>=j+1) )
                stat2[jj,i] = np.sum( self.TOAinfo[i]['TOA'][:,j]>0 )
        df = pd.DataFrame(data=stat, 
                          index=np.arange(nPhone)+1, 
                          columns=np.arange(nTag))
        df2 = pd.DataFrame(data=stat2, 
                          index=self.phoneIDs, 
                          columns=np.arange(nTag))        
        saveCSV=TOA_file+'.csv'
        logger.debug(df)
        logger.debug(df2)
        with open(saveCSV, 'w', encoding="utf-8", newline='') as f:
            f.write("Signal number at each phone \n")
            df.to_csv(f, index=True)
            f.write("\n Detection number at each phone")
            df2.to_csv(f, index=True)
        return df
    @timing
    def creat_one_tag(self,curTag, tol_signal=0.3,badPhones=[],
                      time_start=None, time_end=None,loc_sys=None,
                      ):     
        "build TOA info for a tag"
        info_mat = curTag['decodes'] # date, message time, SNR, phone_ID
        TOAinfo={'TOA':np.empty((0,0)),
                 'TOA_t_num':np.empty(0),
                 'TOA_t_datetime':pd.to_datetime(np.empty(0), unit='D').round('s'),
                 'phoneIDs':np.empty(0),
                 'SNR':np.empty(0),
                 }
        if info_mat is None:
            return TOAinfo
        #tol_snr = 180
        #snr = info_mat[:,2]
        #info_mat = info_mat[snr>tol_snr]
        #print(f"{np.sum(snr<=tol_snr)}/{info_mat.shape[0]} points was removed because of low snr {tol_snr}")
        if len(info_mat)==0:
            return TOAinfo
        phoneIDs=info_mat[:,3]
        ind_keep=np.isfinite(phoneIDs)
        for p in badPhones:
            ind_keep = ind_keep & (np.abs(phoneIDs-p)>1e-3)
        info_mat=info_mat[ind_keep]
        if  len(info_mat)==0:
            return TOAinfo
        
        days = info_mat[:,0]
        #days = np.floor(days)-np.floor(np.min(days))
        t=   info_mat[:,1]*24*3600
        nt = len(t)
        ind = np.argsort(t)
        info_mat = info_mat[ind,:]
        t        = t[ind]
        dt = np.diff(t)
        phoneIDs, ind_phones = np.unique(info_mat[:,3].astype(int),return_inverse=True)
        # build CSR matrix
        indptr   = np.where(dt>tol_signal)[0] + 1
        indptr   = np.hstack( (0, indptr, nt))
        indices  = ind_phones
        shape=(len(indptr)-1, indices.max()+1)                      
        #remove duplicate data
        ind_uni, indices, indptr = self.remove_duplicate_toa(np.arange(len(t)), indices, indptr, shape)
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
    def remove_duplicate_toa(self, data, indices, indptr, shape):
       new_data = []
       new_indices = []
       new_indptr = [0]
       n_rows = shape[0]
       for row in range(n_rows):
           # Get the start and end position of the current row
           start, end = indptr[row], indptr[row + 1]
           row_indices = indices[start:end]
           row_data = data[start:end]
           # Use a dictionary to track seen column indices (keep the first occurrence)
           seen = {}
           for idx, col in enumerate(row_indices):
               if col not in seen:
                   seen[col] = row_data[idx]
           # Sort the results (to maintain sorted column indices)
           sorted_indices = sorted(seen.keys())
           sorted_data = [seen[col] for col in sorted_indices]
           # Update the result
           new_indices.extend(sorted_indices)
           new_data.extend(sorted_data)
           new_indptr.append(len(new_indices))
       return new_data, new_indices, new_indptr  
    
    @classmethod
    def split_one_tagi999(cls, TOAinfo, tol_split=np.inf, testInfo=None, numTrack=10000, dim=2):
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


    @classmethod
    def split_one_tag(cls, TOAinfo, tol_split=np.inf, testInfo=None, numTrack=10000, dim=2):
        All_t_datetime = TOAinfo['TOA_t_datetime']
        TOA = TOAinfo['TOA']
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
            TOA_t_datetime = TOAinfo['TOA_t_datetime'][ind]
            mask = TOA[ind]>1e-3
            valid = np.sum(mask, axis=1)>=dim+1
            nValid = np.sum(valid)
            nDomain = int(np.ceil(nValid/numTrack))
            if nDomain == 0: continue
            numTrack_real = int(mask.shape[0]/nDomain)
            for i in range(nDomain):
                time_range   = TOA_t_datetime[    i*numTrack_real:(i+1)*numTrack_real]
                time_s = time_range[0]
                time_e = time_range[-1]
                intervals.append( (time_s, time_e) )
                names.append(testName+f"_interval{i}")
        return intervals,names




    @timing   
    def apply_TOAD_filter(self, TOAinfo_Tagi, nstd=3,window_size=21):
        # large window_size (21) for more data
        # small window_size (11) for less data
        if TOAinfo_Tagi is None:
            return None
        TOA=TOAinfo_Tagi['TOA']
        if len(TOA)==0:
            return TOAinfo_Tagi
        mask=TOA>0        
        TOA[~mask] = np.nan
        nPhone = TOA.shape[1]
        for k in range(100):
            indictor=np.zeros_like(TOA).astype(int)
            for i in range(nPhone):
                for j in range(i+1,nPhone):                       
                    TDOA = TOA[:,i]-TOA[:,j]
                    if np.all(np.isnan(TDOA)):
                        continue
                    ind_all = np.where(~np.isnan(TDOA))[0]
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
                    ind=np.where(cond)[0]
                    ind=ind_all[ind]
                    indictor[ind,i] += 1;  indictor[ind,j] += 1
            badPoints = (indictor==np.max(indictor, axis=1)[:,None]) & (indictor>0)
            TOA[badPoints] = np.nan
            logger.debug(f"iter:{k}: {np.sum(badPoints)} bad points removed")
            if np.sum(badPoints)==0:
                break
        TOA[np.isnan(TOA)] = -999
        ind_keep=np.any(TOA>0, axis=1)
        TOAinfo_Tagi['TOA'] = TOA[ind_keep]
        TOAinfo_Tagi['TOA_t_num'] = TOAinfo_Tagi['TOA_t_num'][ind_keep]
        TOAinfo_Tagi['TOA_t_datetime'] = TOAinfo_Tagi['TOA_t_datetime'][ind_keep]
        TOAinfo_Tagi['SNR'] = TOAinfo_Tagi['SNR'][ind_keep]        
        return TOAinfo_Tagi

   
                  
    

if __name__=='__main__':
    msgMatFile='drift/combine_all_noDup0.3_drifting_tags.mat'   
    #TOAinfo=TOAinfo_creator(msgMatFile, tol=0.3, update=False)
    msgMatFile = 'McCloud_2023_study/tracking_2D/DriftingTags_timeCorrected.mat'
    badPhones=[2,3]
    msg=loadMsg_Mat(msgMatFile,badPhones=badPhones)
    TOAinfo=TOAinfo_creator(msgMatFile, tol_signal=0.3, update=True, badPhones=badPhones)
    phoneLocCSVFile = 'McCloud_2023_study/McCloud_receiver_locs.csv'
    ss=loadPhone_csv(phoneLocCSVFile,6,0, badPhones=badPhones)
