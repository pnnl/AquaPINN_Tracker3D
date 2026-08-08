# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:25:57 2024

@author: chen096
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.set_loglevel('error')
import networkx as nx
import pickle
from ..SharedLibs.functions import treeToNumpy,treeToTensor,soundSpeed_freshWater
import logging
logger = logging.getLogger(__name__)
from ..SharedLibs.decorators import timing
from .jumpDetection import signal_linear_fit
from .graph import find_best_nodes,check_solvability
from .ransac import ransac
from ..SharedLibs.batchSampler import  BatchSampler

zero= torch.tensor(0.)
one= torch.tensor(1.)


@timing
def getTrainData( TOAinfo,
                  loc_sys,               
                  ignorePhoneNum=1,
                  usePhones=[],
                  ignorePhoneNum_HeadTail=4,
                  badPhones=[],ignoreZ=False,                  
                  ): 
    phoneInfo=loc_sys.phoneInfo
    TOA=[]
    TOA_t_num=[]    
    tagID=[]
    phoneIDs_all=list(phoneInfo['IDs']) #[ i for i in range(1,11) if i not in badPhones]
    beacon_signal_matrix = np.zeros((len(phoneIDs_all), len(phoneIDs_all)))
    invalidPairs=[]#[[1,6],[6,9]]for McCloud
    for iii,(ID, curTOA) in enumerate(TOAinfo.items()):
        tmp=-999*np.ones((curTOA['TOA'].shape[0],len(phoneIDs_all)))
        phoneIDs=list(curTOA['phoneIDs'].astype(int))
        intersect = list(set(phoneIDs_all) & set(phoneIDs))
        ind_ph  = [phoneIDs_all.index(p) for p in intersect]
        ind_toa = [phoneIDs.index(p) for p in intersect ]
        tmp[:,ind_ph] = curTOA['TOA'][:,ind_toa]
        nvalid = np.sum(tmp>0, axis=0)        
        #tmp[:,nvalid<5]=-1
        for pair in invalidPairs:
            if ID in pair:
                tt=pair.copy() 
                tt.remove(ID)
                if tt[0] in phoneIDs_all:
                    col=phoneIDs_all.index(tt[0])
                    tmp[:,col]=-1
        TOA.append( tmp )     
        beacon_signal_matrix[iii,:] = np.sum(tmp>0, axis=0)
        TOA_t_num.append(curTOA['TOA_t_num'] )
        tagID.append(phoneIDs_all.index(ID)*np.ones(curTOA['TOA'].shape[0]).astype(int))
        # phoneIDs.append(list(curTOA['phoneIDs'].astype(int)))
    phoneIDs = phoneIDs_all
    TOA = np.concatenate(TOA, axis=0)
    TOA_t_num = np.concatenate(TOA_t_num, axis=0)
    tagID   = np.concatenate(tagID,axis=0)

    phoneInfo = loc_sys.phoneInfo
    
   
    
    # get phonw locati
    phoneLoc_all= phoneInfo['XYZ']
    phoneIDs_all = list(phoneInfo['IDs'])
        
    phoneLoc = phoneLoc_all[[phoneIDs_all.index(p) for p in phoneIDs]]
    
    
    # remove signals with receiver number smaller than $ignorePhoneNum$
    nvalid = np.sum(TOA>1e-8,axis=1)
    ind= nvalid>1 # ignorePhoneNum is set to 1 for sync
    if np.sum(ind)<=5:
        return None
    
    TOA= TOA[ind]
    TOA_t_num=TOA_t_num[ind]
    tagID=tagID[ind]
    mask   = TOA>1E-8
   
 
    time= ( np.sum(TOA*mask, axis=1)/np.sum(mask,axis=1) )[:,None]    
    data=[time, tagID, TOA, phoneLoc, mask,TOA_t_num]

    return data, phoneIDs                                          


    
    
def draw_graph(adj_matrix, IDs, edge_labels=None,
               title=None, vertexFontSize=15,
               edgeFontSize=12, save=None):
    plt.figure()
    adj_matrix_pseudo = np.ones_like(adj_matrix)
    G_actual= nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    G_pseudo= nx.from_numpy_array(adj_matrix_pseudo, create_using=nx.DiGraph)
    pos = nx.spring_layout(G_pseudo,  seed=42)
    #pos = nx.fruchterman_reingold_layout(G) #(G, k=1/adj_matrix.shape[0], seed=432)
    nx.draw(G_actual, pos, labels=dict(zip(range(len(IDs)), IDs)),
            with_labels=True, node_size=700, node_color='skyblue',
             font_size=vertexFontSize, font_color='black', font_weight='bold', arrows='True')
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(G_actual,pos, edge_labels=edge_labels,
                                     font_size=edgeFontSize, font_color='red')
    if title is not None:
        plt.title(title)
    # plt.show()
    if save is not None:
        plt.savefig(save, dpi=300) 
    plt.close()
    
    
def analyzePairDetection(time, toa_phoneOnbeacon, toa_phone2, travelTime,
                         mask_phoneOnbeacon, mask_phone2,
                         threshold_detectionNum=5,
                         noise_tolerance=8/1500,
                         window_size=5, max_median_ratio=10,
                         ):
    # assume  the dectction from a phone-atteched beacon to a phone is linealy ralted with time
    # errFlag>=1: success
    #        =-1: no detections on the phone with attached beacon
    #        =-2: no detections on the other phone
    #        =-3: no enough parired signals 
    #        =-4: the linear assumption does not hold
    ind_jump_x=None
    jump_y = None
    ind_inlier=None
    ind_outlier=None
    errFlag = 1 
    msg  = 'paired successfully'
    if np.sum(mask_phoneOnbeacon)<threshold_detectionNum:
        errFlag = -1 
        msg= 'no detections on the phone with attached beacon'
    if np.sum(mask_phone2)<threshold_detectionNum:
        errFlag = -2 
        msg= 'no detections on the second phone'
    mask_all = mask_phoneOnbeacon&mask_phone2
    
    ind_all = np.arange(len(mask_all))
    tdoa = toa_phone2-toa_phoneOnbeacon
    if np.sum(mask_all)<threshold_detectionNum:
        errFlag = -3 
        msg = 'no enough parired detections'
    if errFlag<0:
        return errFlag, msg, None,(None, None, None, None, None)
    signal_x = time[mask_all]
    signal_y = tdoa[mask_all, None]-travelTime
    ind_jump_x, fitted_function, jump_y = signal_linear_fit(signal_x, signal_y, 
                                                            threshold=noise_tolerance, 
                                                            window_size=window_size, max_median_ratio=max_median_ratio,
                                                            threshold_signalNum=threshold_detectionNum)
    
    pred_y= fitted_function(signal_x)
    residuals = np.abs(signal_y - pred_y)
    inliers = residuals < noise_tolerance
    inliers_remove= residuals < 2*noise_tolerance
    ind_inliers =  ind_all[mask_all][inliers_remove.ravel()]
    ind_outlier =  ind_all[mask_all][~inliers_remove.ravel()]
    inliers_count = np.sum(inliers)    
    inliers_ratio = inliers_count/signal_y.size
    logger.info(f'inliers_ratio={inliers_ratio}')    
    
    if inliers_ratio<0.4:
        errFlag = -4 
        msg = f'the linear assumption does not hold, inlier ratio is {inliers_ratio}'
    if ind_jump_x is not None:
        t1 = toa_phoneOnbeacon[mask_all]
        t2 = toa_phone2[mask_all]
        jump_x_phoneOnbeacon = 0.5* ( t1[ind_jump_x]+t1[ind_jump_x-1])
        jump_x_phone2        = 0.5* ( t2[ind_jump_x]+t2[ind_jump_x-1])
    return errFlag, msg, fitted_function, (jump_x_phoneOnbeacon,jump_x_phone2 , jump_y, ind_inliers,ind_outlier)    
    

## physics-based network model   
class Model(torch.nn.Module):
    def __init__(self, fileSystem, data, dam='LMN',err_GPS=10,
                 temperature_ref=15, dis_detect=100,
                 region_lb=[0,0,0], region_ub=[1,1,1],
                 refID=None,
                 ignoreZ=False,
                 toSolveLoc=True,
                 toSolveSync=True,
                 distance_beacon_phone=0.2,
                 batchSize=50000,
                 ):
        '''        

        Parameters
        ----------
        fileSystem : class
            file name system.
        data : tuple
            data generated from getTrainData.
        temperature_ref : float, optional
            reference temperature. The default is 15.
        dis_detect : float, optional
            maximum detection distance of hydrophone. The default is 100.
        region_lb, region_ub : np.ndarray, optional
            lower and upper box bound osf the region     
            The default is 1.
        ignoreZ : bool, optional
            the third dimension is ignored if ignoreZ is True. The default is 5.            
        Returns
        -------
        None.

        '''        
        super(Model, self).__init__()
        self.toSolveSync=toSolveSync
        self.toSolveLoc=toSolveLoc
        self.FNS=fileSystem
        self.dam = dam
        self.err_GPS=err_GPS
        self.dis_detect =dis_detect
        self.dim = 3 if not ignoreZ else 2
        # load data
        self.data = data[0]
        self.phoneIDs=data[1]
        self.soundSpeed_ref = soundSpeed_freshWater(temperature_ref)
        self.region_lb=treeToTensor(np.array(region_lb)).reshape(1,3)
        self.region_ub=treeToTensor(np.array(region_ub)).reshape(1,3)
        t,tagID, toa_noisy, phoneLoc, mask, TOA_t_num=self.data
        self.ind_refID = self.phoneIDs.index(refID) if refID is not None else None
        self.ind_problematicPhones=[]
        # define parameter
        self.initNet(t,toa_noisy)
        self.oneVec=torch.ones(phoneLoc.shape[0])
        self.oneVec[self.ind_refID] = 0
        self.distance_beacon_phone=distance_beacon_phone
        self.weight=torch.ones(toa_noisy.shape)
        self.batchSize=min(batchSize,t.shape[0])
        self.sampler = BatchSampler(t.shape[0], batchSize=self.batchSize)
        # temp modification
        #tagID = tagID.astype(int)
        #ind = np.arange(len(tagID))
        #mask[ind, tagID]=False
        #self.data[-2] =mask       
        return
        
    def initNet(self, t, toa_noisy):
        '''
        define the network and 
        other training parameters
        ----------
        top : time of ping for each signal
        ss  : sound speed 
        toa_tau : variance of toa for each phone
        '''
        Nt, Nhp = toa_noisy.shape                
        self.Npiece = Npiece  = np.maximum(int((t.max()-t.min())/24/3600),1) 
        shift_polyVec=torch.nn.Parameter(torch.zeros(Nhp, 3, Npiece))
        phoneLocReal =torch.nn.Parameter(torch.zeros(Nhp,3))
        beaconPhoneDis = torch.nn.Parameter(torch.zeros(Nhp))
        top=torch.nn.Parameter(torch.zeros(Nt,1))
        
        toa_tau  = torch.nn.Parameter(torch.zeros(1,Nhp))

        self.tmin, self.tmax=t.min().item(),t.max().item()
        self.t_norm=lambda t: ( t - (self.tmax+self.tmin)/2 )/( (self.tmax-self.tmin)/2 )
        t_norm = self.t_norm(toa_noisy)
        ind_piece = (t_norm+1)/2*Npiece
        ind_piece = np.maximum(np.minimum(ind_piece, Npiece-1e-3), 0)
        self.ind_piece = treeToTensor(ind_piece).to(torch.int)
        
        errFlagMatrix,  pairs = self.analyzeData()
        print("errFlag", errFlagMatrix)
        refID, syncable_phoneIDs = self.analyzeValidSync(errFlagMatrix)
        if self.ind_refID is None:            
            self.ind_refID = self.phoneIDs.index(refID)
        else:
            logger.info(f'refID {self.phoneIDs[self.ind_refID]} has been set by user')
        
        self.syncable_phoneIDs = syncable_phoneIDs
        
        t_jump = -999*torch.ones(Nt,Nhp)
        for pair in pairs:
            beacon_ID, phone_ID, jump_x_beacon, jump_x_phone, ind_inlier, ind_outlier = pair
            beacon_ind=self.phoneIDs.index(beacon_ID)
            phone_ind =self.phoneIDs.index(phone_ID)
            if (beacon_ID in syncable_phoneIDs) and (phone_ID in syncable_phoneIDs):
                n_jump = len(jump_x_beacon)
                t_jump[:n_jump,beacon_ind] = treeToTensor(self.t_norm(jump_x_beacon))
                t_jump[:n_jump,phone_ind] =  treeToTensor(self.t_norm(jump_x_phone))
        
        t_jump_share = torch.zeros(0)
        for i in range(Nhp):
            t_jump_share = torch.cat((t_jump_share,t_jump[t_jump[:,i]>-10,i]))
        Njump = len(t_jump_share)
        t_jump[:Njump]=t_jump_share[:,None]
            
        t_jump_all = []
        value_jump_all =[]            
        for i in range(Nhp):
            t_jump_i = t_jump[t_jump[:,i]>-10,i]
            t_jump_all.append(torch.sort(t_jump_i)[0])
            value_jump_i = torch.nn.Parameter(torch.zeros_like(t_jump_i))
            value_jump_all.append(value_jump_i)
        stat = [toa_tau,]
        sol = [top, phoneLocReal, beaconPhoneDis, shift_polyVec, ]
        
        self.paras = [sol , stat, t_jump_all, value_jump_all]        
        
        return    
    
    def analyzeData(self):
        t,beacondIND, toa_noisy, phoneLoc, mask, TOA_t_num=self.data
        mask=np.bool_(mask)
        beacondIND=beacondIND.astype(int)
        
        phoneIDs = self.phoneIDs
        nPhone = len(phoneIDs)
        errFlagMatrix=np.ones((nPhone, nPhone))
        pairs=[]    
        for beacon_ind in range(nPhone):
            for phone_ind in range(nPhone):
                beaconID, phoneID = phoneIDs[beacon_ind], phoneIDs[phone_ind]
                if phone_ind==beacon_ind:
                    continue
                ind = beacondIND==beacon_ind 
                ind_number = np.arange(len(ind))
                travelTime= np.linalg.norm(phoneLoc[phone_ind,:self.dim]-phoneLoc[beacon_ind,:self.dim])/self.soundSpeed_ref
                
                err, msg, fitted_function, (jump_x_beacon,jump_x_phone , jump_y, ind_inlier, ind_outlier) = analyzePairDetection(self.t_norm(t[ind]),
                                                             toa_noisy[ind,beacon_ind], 
                                                             toa_noisy[ind,phone_ind], travelTime,
                                                           mask[ind,beacon_ind], mask[ind,phone_ind])
                
                
                errFlagMatrix[beacon_ind,phone_ind] = err
                logger.info(f'Pair detection status for beacon {beaconID}->phone {phoneID}: {msg}')                
                if err>=0:
                    pair =[phoneIDs[beacon_ind], phoneIDs[phone_ind], jump_x_beacon, jump_x_phone, 
                           ind_number[ind][ind_inlier], ind_number[ind][ind_outlier]]
                    pairs.append(pair)
        return errFlagMatrix,  pairs

    def analyzeValidSync(self, errFlagMatrix):
        t,beacondIND, toa_noisy, phoneLoc, mask, TOA_t_num=[ treeToNumpy(p) for p in self.data]
        phoneIDs = self.phoneIDs
        # reference phone with maximum connections
        dis = phoneLoc[:,None,:self.dim]-phoneLoc[None,:,:self.dim]
        dis = np.linalg.norm(dis, axis=2)
        connectMatrix = errFlagMatrix.copy()
        connectMatrix[errFlagMatrix<0]=0
        connectMatrix[errFlagMatrix>=0]=1
        ind_refID = find_best_nodes(connectMatrix, dis=dis)
        solvability = check_solvability(connectMatrix, ind_refID)
        
        refID=phoneIDs[ind_refID]
        logger.info(f'Phone {refID} is chosen as reference for sync')
        
        syncable_phoneIDs_sync=[phoneIDs[k] for k,v in solvability.items() if v]
        free_phoneIDs_sync=[phoneIDs[k] for k,v in solvability.items() if not v]
        logger.info(f"time of phones can be synced: {syncable_phoneIDs_sync}")
        logger.info(f"time of phones can not be synced: {free_phoneIDs_sync}")        
        return refID, syncable_phoneIDs_sync
    
        

    
   
    
    def initOpt(self, lr=1e-3, lr_lambda=lambda step:0.99**(step/400) ):
        '''        
        initialize optimizers
        Parameters
        ----------
        lr : float, optional
            learning rate. The default is 1e-3.
        lr_lambda : function, optional
            updating rule of learning rate. The default is lambda step:1.
        '''
        # define optimizer
        self.opt_Adam    = torch.optim.Adam(self.paras[0] + self.paras[1] + self.paras[3], 
                                            lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR( self.opt_Adam, lr_lambda=lr_lambda )   
        self.opt_BFGS = torch.optim.LBFGS(
            self.paras[0] + self.paras[1] + self.paras[3],
            lr=1.0,
            max_iter=20,
            max_eval=25,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            history_size=100,
            line_search_fn='strong_wolfe',  # enables proper curvature-guided step sizes
        )     


    def getHubLoss(self, noise_toa, delta):
        noise_toa = torch.abs(noise_toa)
        mask_small = noise_toa<=delta
        loss_pos_small  = 0.5*noise_toa**2 #)/2*(1/3)**2).mean()
        loss_pos_large  = delta*(noise_toa-0.5*delta) #0.5*(torch.square(xyz_LF[:, :self.dim]-xyz_LF_NN[:, :self.dim])/2*(1/3)**2).mean()
        loss_pos = loss_pos_small*mask_small + (~mask_small)*loss_pos_large                        
        return loss_pos

        
    def lossVec0(self, paras, data, indices):
        "define loss matrix for assessment"
        phoneLocReal,beaconPhoneDis, top, toa, shift_polyVec, stat, t_jump_all, value_jump_all = self.decodeInput(paras, data, indices)
        toa_tau,  =stat     
        t, tagID,toa_noisy, phoneLoc, mask, TOA_t_num=data
        mask = mask.to(torch.bool)
        tagID = tagID.to(torch.int)
        
        # position constriant
        loc= phoneLocReal if self.toSolveLoc else phoneLoc
        dis = loc[tagID,None,:self.dim]-loc[None,:,:self.dim]
        r   = torch.norm(dis, dim=2) + torch.diag(beaconPhoneDis)[tagID]
        noise_toa = top+r/self.soundSpeed_ref-toa
        with torch.no_grad():
            dt= 2/self.soundSpeed_ref
            noise_ratio = torch.minimum( torch.abs(noise_toa)//(dt), 10*one )
            self.weight = 1/noise_ratio*toa_tau[0,0]/torch.max(toa_tau[0,0])*torch.ones_like(noise_toa)
            weightMask = (torch.abs(noise_toa)>dt)&mask
            self.weight[~weightMask] = 1    
        loss_pos = (self.weight**2*noise_toa.square())[mask].sum()/(self.weight**2)[mask].sum()
        return loss_pos,
    
    def lossVec(self, paras, data, indices):
        "define loss matrix for traning"
        phoneLocReal,beaconPhoneDis, top, toa, shift_polyVec,stat , t_jump_all, value_jump_all   = self.decodeInput(paras, data, indices)
        toa_tau,  =stat     
        t, tagID,  toa_noisy,phoneLoc, mask, TOA_t_num=data
        mask = mask.to(torch.bool)
        tagID = tagID.to(torch.int)        
        # position constriant
        loc= phoneLocReal if self.toSolveLoc else phoneLoc
        dis = loc[tagID,None,:self.dim]-loc[None,:,:self.dim]
        r   = torch.norm(dis, dim=2) + torch.diag(beaconPhoneDis)[tagID]
        noise_toa = top+r/self.soundSpeed_ref-toa
        with torch.no_grad():
            dt= 2/self.soundSpeed_ref
            noise_ratio = torch.minimum( torch.abs(noise_toa)//(dt), 10*one )
            self.weight = 1/noise_ratio*toa_tau[0,0]/torch.max(toa_tau[0,0])*torch.ones_like(noise_toa)
            weightMask = (torch.abs(noise_toa)>dt)&mask
            self.weight[~weightMask] = 1              
        # loss_pos = (self.weight**2*torch.square(noise_toa)/2*toa_tau[0,0]**2-torch.log(toa_tau[0,0]))[mask].mean()       
        delta = 3/self.soundSpeed_ref
        loss_pos=self.getHubLoss( noise_toa, delta)
        tao = toa_tau[0,0]
        term1 = 1/tao*np.sqrt(np.pi/2)*torch.erf(delta/np.sqrt(2)*tao)
        term2 = 1/tao**2/delta*torch.exp(-delta**2/2*tao**2)
        loss_pos = (loss_pos*tao**2+torch.log(term1+term2))[mask].sum()

        return loss_pos, 

    def loss_func(self, paras, data, indices, BFGS=False, weight_decay=0):
        "define loss fuction"
        if not BFGS:
            self.opt_Adam.zero_grad()
        else:
            self.opt_BFGS.zero_grad()        
        lossVec = self.lossVec(paras, data, indices)
        loss_all=torch.cat( [ k.flatten() for k in lossVec ] )
        loss = loss_all.sum()
        loss.backward()      
        # torch.nn.utils.clip_grad_value_(self.paras[:-1] + self.paras[-1], clip_value=1E6)
        return loss
    def getBatchDataTensor(self, data, indices):
        #data=[time, tagID, TOA, phoneLoc, mask,TOA_t_num]
        time, tagID, TOA, phoneLoc, mask,TOA_t_num = data
        newdata = treeToTensor([time[indices], tagID[indices], TOA[indices], 
                                phoneLoc, mask[indices], TOA_t_num[indices]])
        return newdata, treeToTensor(indices).to(torch.int)
    
    def decodeInput(self, paras, data, indices):
        "decode network training parameters"
        sol, stat, t_jump_all, value_jump_all,=paras
        top, phoneLocReal,beaconPhoneDis, shift_polyVec=sol
        toa_tau,  = stat
        t, tagID, toa_noisy, phoneLoc, mask, TOA_t_num= data
        fun = lambda x: torch.tanh(x)  #lambda x: torch.clamp(x,-1,1)
        beaconPhoneDis = torch.sigmoid(beaconPhoneDis)*10 # 1m distance
        
        phoneLocReal = fun(phoneLocReal)*self.err_GPS + phoneLoc
        phoneLocReal[self.ind_refID] = phoneLoc[self.ind_refID]
        for i in self.ind_problematicPhones:
            phoneLocReal[i] = phoneLoc[i]
        #phoneLocReal = phoneLoc+0
        top = t+15*fun(top[indices])
        fun2 = lambda x,n: 1+torch.exp(-5*x/n)
        toa_tau  = fun2(toa_tau,1)                 
        stat = [ toa_tau,]
        
        t_norm = self.t_norm(toa_noisy)
        ind_piece = self.ind_piece[indices]
        shift_polyVec = 10*fun(shift_polyVec)*self.oneVec[:,None,None]
        #shift_polyVec2 = torch.einsum('imni->imn', shift_polyVec[:,:,ind_piece])
        shift_polyVec2 = shift_polyVec[torch.arange(toa_noisy.shape[1])[None,:], :, ind_piece]
        a=shift_polyVec2[:,:,0]
        b=shift_polyVec2[:,:,1]
        c=shift_polyVec2[:,:,2]       

        
        toa = toa_noisy + b*t_norm+c + a*t_norm**2 
        #toa = toa_noisy + b.T*t_norm+c.T
        value_jump_all = [15*torch.tanh(p) for p in value_jump_all]
        for i in range(toa.shape[1]):
            if len(value_jump_all[i])<=0: continue
            value_jump_i=torch.cat((value_jump_all[i][:1], torch.diff(value_jump_all[i])), dim=0 )
            t_jump_i=t_jump_all[i]
            toa_jump = (t_norm[:,i,None]>t_jump_i[None,:])*value_jump_i[None,:]
            toa[:,i]+= torch.sum(toa_jump, dim=-1)*self.oneVec[i]
        return phoneLocReal, beaconPhoneDis, top, toa, shift_polyVec, stat, t_jump_all, value_jump_all
    
    @timing
    def train(self, nIterAdam, nIterBFGS,
              lr, lr_lambda,
              displayAdam=100, displayBFGS=10,
              problematicPhones=[],
              patience=15):
        '''
        Parameters
        ----------
        nIterAdam : int
            Number of Adam mini-batch iterations.
        nIterBFGS : int
            Number of full-batch L-BFGS iterations run after Adam.
            L-BFGS uses second-order curvature and strong Wolfe line search
            so it converges the residual drift far faster than Adam.
            Recommended: 100-300.  0 disables.
        lr : float
            Adam learning rate.
        lr_lambda : function
            LR schedule for Adam.
        displayAdam : int, optional
            Log status every this many Adam epochs.  Default 100.
        displayBFGS : int, optional
            Log status every this many L-BFGS epochs.  Default 10.
        problematicPhones : list
            Phone IDs whose location is held fixed.
        patience : int, optional
            Adam early-stop patience in units of displayAdam epochs (i.e.
            patience * displayAdam Adam steps without improvement triggers
            early stop).  Default 15 (= 1500 Adam steps).
        Returns
        -------
        None.
        '''
        self.ind_problematicPhones=[self.phoneIDs.index(p) for p in problematicPhones]
        self.initOpt(lr=lr, lr_lambda=lr_lambda)
        error_RMSE_min, loss_likelihood_min = 1e10, 1e10
        no_improve_count = 0
        for epoch in range(nIterAdam):
            # mini-batch Adam step
            indices = self.sampler.get_next()
            data, indices = self.getBatchDataTensor(self.data, indices)
            self.loss_func(self.paras, data, indices, BFGS=False)
            self.opt_Adam.step()
            self.lr_scheduler.step()
            if epoch == 0 or (epoch + 1) % displayAdam == 0:
                error_RMSE, loss_likelihood = self.record_status('Adam', epoch)
                improved = (error_RMSE < error_RMSE_min * 0.9995
                            or loss_likelihood < loss_likelihood_min * 0.9995)
                if improved:
                    error_RMSE_min = min(error_RMSE.item(), error_RMSE_min)
                    loss_likelihood_min = min(loss_likelihood.item(), loss_likelihood_min)
                    self.saveParas(status='best')
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                logger.info(f'  patience: {no_improve_count}/{patience}')
                if no_improve_count >= patience:
                    logger.info(
                        f'Early stopping at Adam epoch {epoch + 1}: '
                        f'no improvement for {patience * displayAdam} steps.')
                    break
        # ----------------------------------------------------------------
        # Full-batch L-BFGS polishing
        # Runs on the entire dataset; strong Wolfe line search provides
        # reliable convergence for the smooth Huber loss.
        # Bug-fix: was incorrectly calling opt_Adam here — now opt_BFGS.
        # Bug-fix: loss_func requires indices; pass all indices for full-batch.
        # ----------------------------------------------------------------
        if nIterBFGS > 0:
            Nt = self.data[0].shape[0]
            all_indices = np.arange(Nt)
            data_full, indices_full = self.getBatchDataTensor(self.data, all_indices)
            for epoch in range(nIterBFGS):
                self.opt_BFGS.step(
                    lambda: self.loss_func(
                        self.paras, data_full, indices_full, BFGS=True))
                if epoch == 0 or (epoch + 1) % displayBFGS == 0:
                    error_RMSE, loss_likelihood = self.record_status('BFGS', epoch)
                    if (error_RMSE.item() < error_RMSE_min * 0.9995
                            or loss_likelihood.item() < loss_likelihood_min * 0.9995):
                        error_RMSE_min = min(error_RMSE.item(), error_RMSE_min)
                        loss_likelihood_min = min(loss_likelihood.item(), loss_likelihood_min)
                        self.saveParas(status='best')
        self.saveParas()
        self.plot_train_history()
        return
    def record_status(self,optimizer, epoch, ):
        nt = self.data[0].shape[0]
        indices=np.random.permutation(nt)[:4*self.batchSize]
        data,indices = self.getBatchDataTensor(self.data, indices)
        error_RMSE,     = self.lossVec0(self.paras, data, indices)
        loss_likelihood, = self.lossVec(self.paras, data, indices)
        
        line = optimizer+', %d, %.3e, %.3e' % (epoch,
                                                loss_likelihood.item(),
                                                error_RMSE.item(),
                                                )
        logger.info(line) 
        writeMode = "w+" if epoch==0 else "a+"
        with open(self.FNS.history(), writeMode) as f:
            if epoch==0:
                header = 'Optimizer,Epoch,loss_likelihood,error_RMSE'
                f.write(header+'\n')
            f.write(line+'\n')
        #paras=self.decodeInput(self.paras, self.data)        
        #stat = paras[-3]
        #logger.info(f"{treeToNumpy(stat)}")
        return error_RMSE, loss_likelihood
    
    def plot_train_history(self):
        import matplotlib.pyplot as plt
        df=pd.read_csv(self.FNS.history())        
        epoch = df['Epoch'].values
        error_RMSE = df['error_RMSE'].values
        loss_likelihood = df['loss_likelihood'].values
        fig,ax1=plt.subplots(figsize=(8, 6))        
        ax1.plot(epoch, loss_likelihood,'-r', label='likelihood')
        ax2 = ax1.twinx()
        ax2.semilogy(epoch, error_RMSE, label='RMSE')
        lines1,labels1=ax1.get_legend_handles_labels()
        lines2,labels2=ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, loc='best')
        ax1.set_ylabel('likelihood')
        ax2.set_ylabel('RMSE')
        plt.savefig(self.FNS.historyfig(), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def saveParas(self, netfile=None, status='normal'):
        "save checkpoint file"
        netfile = netfile or (self.FNS.net() if status!='best' else self.FNS.net('best'))        
        torch.save({'paras': self.paras,
                   },netfile,
                    )
        return 
    
    def loadParas(self, netfile=None, status='normal'):
        "load from checkpoint file"
        netfile = netfile or (self.FNS.net() if status!='best' else self.FNS.net('best'))
        info = torch.load(netfile, map_location=lambda storage, loc: storage)
        paras=info['paras']
        self.paras =(  [[p.to(torch.defaultDevice) for p in paras[0] ]] 
                     + [[p.to(torch.defaultDevice) for p in paras[1]]]
                     + [[p.to(torch.defaultDevice) for p in paras[2]]]
                     + [[p.to(torch.defaultDevice) for p in paras[3]]]
                     )    
        return
    def summary(self, syncFile):
        indices=np.array([0]).astype(int)
        data,indices = self.getBatchDataTensor(self.data, indices)           
        phoneLocReal,beaconPhoneDis, _, _, shift_polyVec,stat, t_jump_all, value_jump_all    = self.decodeInput(self.paras, data, indices)   
        t, tagID,  toa_noisy,phoneLoc, mask, TOA_t_num=self.data               
        # check phone shift
        problematicPhones=[]
        for i in range(phoneLocReal.shape[0]):
            dis= torch.norm(phoneLocReal[i]-treeToTensor(phoneLoc)[i])
            if dis>self.err_GPS*0.9:
                logger.warning(f"the location shift ({dis}) of phone #{self.phoneIDs[i]} is too large, the correction is rejected")
                phoneLocReal[i]=phoneLoc[i]    
                problematicPhones.append(self.phoneIDs[i])
                
        shift_polyVec=treeToNumpy(shift_polyVec)
        t_jump_all     = [treeToNumpy(p) for p in t_jump_all    ]
        value_jump_all = [treeToNumpy(p) for p in value_jump_all]
                        
        timeSync = { ID: (shift_polyVec[i],t_jump,v_jump )for i,(ID,t_jump,v_jump) in enumerate(zip(self.phoneIDs, t_jump_all, value_jump_all)) }                             
        with open(syncFile, "wb") as f:
            pickle.dump({'timeSync':timeSync,
                         'phoneIDs':self.phoneIDs,
                         "phoneLoc":treeToNumpy(phoneLocReal),
                         "syncable_phoneIDs":self.syncable_phoneIDs,
                         'tmin':self.tmin,
                         'tmax':self.tmax}, f)

        logger.debug(f'old={phoneLoc}')
        logger.debug(f'new={treeToNumpy(phoneLocReal)}')
        logger.debug(f'shift={treeToNumpy(phoneLocReal)-phoneLoc}')
        logger.debug(f'beaconPhoneDis={treeToNumpy(beaconPhoneDis)}')
        
        logger.debug(f'shift_polyVec={shift_polyVec}')
        return problematicPhones                   
    def plot(self, syncFile=None):        
        nt = self.data[0].shape[0]
        indices=np.arange(nt)
        data,indices = self.getBatchDataTensor(self.data, indices)
        _,_, top, toa, _,_, _, _    = self.decodeInput(self.paras, data, indices)
        top, toa=treeToNumpy((top, toa))
        #toa_tau,  =stat     
        t, tagID,  toa_noisy,phoneLoc, mask, TOA_t_num=self.data
        mask = mask.astype(np.bool_)
        tagID = tagID.astype(int)
        timestamps = pd.to_datetime(treeToNumpy(TOA_t_num), unit='D').round('s').values
        coeff_sync=dict()
        if syncFile is not None:
            with open(syncFile, 'rb') as f:
                syncInfo = pickle.load(f)         
            coeff_sync = syncInfo['timeSync']   
            tmax, tmin  = syncInfo['tmax'], syncInfo['tmin']
            tnorm = lambda t: (t-(tmax+tmin)/2)/(tmax-tmin)*2
        # position constriant
        dis = phoneLoc[:,None,:self.dim]-phoneLoc[None,:,:self.dim]
        rPhone  = np.linalg.norm(dis, axis=2)        
        dis = phoneLoc[tagID,None,:self.dim]-phoneLoc[None,:,:self.dim]
        r  = np.linalg.norm(dis, axis=2)
        noise_toa = r/self.soundSpeed_ref-toa_noisy
        tmin_plot = toa_noisy[mask].min()
        tmax_plot = toa_noisy[mask].max()
        for i in range(toa_noisy.shape[1]):
            for j in range(i+1,toa_noisy.shape[1]):
                id_i,id_j=self.phoneIDs[i], self.phoneIDs[j]
                syncable_i = 'T' if id_i in self.syncable_phoneIDs else 'F'
                syncable_j = 'T' if id_j in self.syncable_phoneIDs else 'F'                
                if np.sum(mask[:,i])<=0 or np.sum(mask[:,j])<=0:
                    continue
                indi, indj = (tagID==i) & mask[:,j], (tagID==j)*mask[:,i]
                ddi = (toa[indi,j] - top[indi,0])*self.soundSpeed_ref
                ddj = (toa[indj,i] - top[indj,0])*self.soundSpeed_ref
                plt.figure()
                plt.plot(timestamps[indi],ddi,'r.', label=f'{id_i}', markersize=2)
                plt.plot(timestamps[indj],ddj,'g.', label=f'{id_j}', markersize=2)                  
                plt.axhline(y=rPhone[i,j], color='k', linestyle='-')
                plt.legend(loc='best')
                filename=self.FNS.figTrack(f"disPair{id_i}_{id_j}_{syncable_i}_{syncable_j}")
                plt.savefig(filename, dpi=300)
                plt.close()
                
                
                dt = noise_toa[:,j]-noise_toa[:,i]                   
                plt.figure()                
                ind = (mask[:,i]&mask[:,j])
                dt = dt[ind]
                t  = timestamps[ind]    
                labels=np.array(self.phoneIDs)[tagID[ind]]
                if len(t)<5:
                   continue
                scatter = plt.scatter(t,dt, c=labels, cmap=plt.get_cmap('tab10'), s=2)
                plt.legend(*scatter.legend_elements(), loc='best')
                
                if coeff_sync is not None:
                    if id_i in coeff_sync and id_j in coeff_sync:
                        (ai,bi,ci), ti_jump, vi_jump=coeff_sync[id_i]
                        (aj,bj,cj), tj_jump, vj_jump=coeff_sync[id_j]   
                        vi_jump=np.concatenate((vi_jump[:1], np.diff(vi_jump)), axis=0 )
                        vj_jump=np.concatenate((vj_jump[:1], np.diff(vj_jump)), axis=0 )
                        tPair=np.linspace(tmin_plot, tmax_plot,1000)
                        tPair_norm = tnorm(tPair)
                        Npiece=len(ai)
                        ind_piece = (tPair_norm+1)/2*Npiece
                        ind_piece = np.maximum(np.minimum(ind_piece, Npiece-1e-10), 0).astype(np.int32)
                        ai,bi,ci=[p[ind_piece] for p in [ai, bi, ci]]
                        aj,bj,cj=[p[ind_piece] for p in [aj, bj, cj]]                          
                        ti_shift=ai*tPair_norm**2+ bi*tPair_norm + ci
                        tj_shift=aj*tPair_norm**2+ bj*tPair_norm + cj
                        toai_jump = (tPair_norm[:,None]>ti_jump[None,:])*vi_jump[None,:]
                        toaj_jump = (tPair_norm[:,None]>tj_jump[None,:])*vj_jump[None,:]
                        ti_shift += toai_jump.sum(axis=-1)
                        tj_shift += toaj_jump.sum(axis=-1)
                        tPair_shift = tj_shift-ti_shift  
                        timestamps_pair = pd.to_datetime(tPair/24/3600-719529, unit='D').round('s').values
                        plt.plot(timestamps_pair,tPair_shift,'k-')                        
                plt.title(f"({id_i},{id_j})-({syncable_i}, {syncable_j})")
                # plt.legend(loc='best')
                filename=self.FNS.figTrack(f"timePair{id_i}_{id_j}_{syncable_i}_{syncable_j}")
                plt.savefig(filename, dpi=300)
                plt.close()
        

        
        
    def showGraph(self, loc_sys, syncFile=None):
        time, tagID, TOA, phoneLoc, mask,TOA_t_num=self.data
        phoneIDs=loc_sys.phoneInfo['IDs']
        beacon_signal_matrix = np.zeros((len(phoneIDs), len(phoneIDs)))
        for iii in range(len(phoneIDs)):
            tmp=TOA[tagID==iii]
            beacon_signal_matrix[iii,:] = np.sum(tmp>0, axis=0)        
        adj_matrix=beacon_signal_matrix.astype(int)
        adj_matrix[adj_matrix>0]=1
        np.fill_diagonal(adj_matrix,0)

        indices=np.array([0]).astype(int)
        data,indices = self.getBatchDataTensor(self.data, indices)
        phoneLocReal,beaconPhoneDis, _, _, shift_polyVec,stat , t_jump_all, value_jump_all   = self.decodeInput(self.paras, data, indices)   

        phoneDisRef = np.linalg.norm(phoneLoc[:,None,:]-phoneLoc[None,:,:], axis=-1)
        phoneDisReal = torch.norm(phoneLocReal[:,None,:]-phoneLocReal[None,:,:], dim=-1)
        phoneDisReal = treeToNumpy(phoneDisReal)

        edge_signalNumber = {}
        edge_phoneDis={}
        for i in range(beacon_signal_matrix.shape[0]):
            for j in range(beacon_signal_matrix.shape[1]):
                if i>j and (beacon_signal_matrix[i,j]>0 or beacon_signal_matrix[j,i]>0):                    
                    edge_signalNumber[(i,j)]= f"{beacon_signal_matrix[i,j]:.0f},{beacon_signal_matrix[j,i]:.0f}"                    
                    edge_phoneDis[(i,j)]=f"{phoneDisRef[i,j]:.2f},{phoneDisReal[j,i]:.2f}"
        file_signalNumber=self.FNS.figTrack(f"signalNumber")                
        file_timeShift=self.FNS.figTrack(f"timeShift") 
        file_phoneDis=self.FNS.figTrack(f"phoneDis") 
        draw_graph(adj_matrix, phoneIDs, edge_signalNumber,title='Signal number',save=file_signalNumber)
        draw_graph(adj_matrix, phoneIDs, edge_phoneDis,title='Phone distance', save=file_phoneDis)

        shift_polyVec=treeToNumpy(shift_polyVec.mean(axis=-1))
         
        
        shift=[ f'{i}: \n'+'%.2e,%.2e,%.2e'%tuple(p) for i, p in zip(phoneIDs,shift_polyVec)]
        draw_graph(adj_matrix, shift,title='Sync shift',vertexFontSize=9, save=file_timeShift)
                
   
        phoneLoc=treeToNumpy(phoneLoc)
        phoneLocReal=treeToNumpy(phoneLocReal)            
        plt.figure()
        plt.scatter(phoneLoc[:,0], phoneLoc[:,1], color='red')
        plt.scatter(phoneLocReal[:,0], phoneLocReal[:,1], color='green')
        for i in range(len(phoneLoc)):
            id_i=phoneIDs[i]
            shift =phoneLocReal[i] - phoneLoc[i]
            str_shift=f'[{shift[0]:0.2f}, {shift[1]:0.2f}, {shift[2]:0.2f}]'
            plt.text(phoneLoc[i,0], phoneLoc[i,1], f'{id_i}\n\n'+str_shift, fontsize=12,
                     ha='center',va='center')
            plt.text(phoneLocReal[i,0], phoneLocReal[i,1], f'{id_i}', fontsize=12,
                     ha='center',va='center')
        plt.xlabel('X')
        plt.ylabel('Y')
        filename=self.FNS.figTrack("PhoneLoc")
        plt.savefig(filename, dpi=300)
        plt.close()
        
        return 
