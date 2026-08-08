# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:25:57 2024

@author: chen096
"""

import torch
import numpy as np
import pandas as pd
import pickle
from ..SharedLibs.networks import MFNN, BSplineModel3D
from ..SharedLibs.decorators import timing
from ..SharedLibs.functions import soundSpeed_freshWater
import logging
logger = logging.getLogger(__name__)



zero= torch.tensor(0.) 
one= torch.tensor(1.)
# transformation between numpy as tensor
Tensor2Numpy= lambda x: x.detach().cpu().numpy()
Numpy2Tensor= lambda x: torch.as_tensor(x)

def getTrainData( TOAinfo,
                  phoneLocCSVFile=None,
                  phoneInfo=None,
                  ignorePhoneNum=1,
                  ignorePhoneNum_HeadTail=4,
                  ignoreZ=False,dam='', numTrack=8000000): 
    minPhoneNum = 4 if not ignoreZ else 3 
   
    phoneLoc_all= phoneInfo['XYZ']
    phoneIDs_all = list(phoneInfo['IDs'])

 
    # get TOA
    TOA=TOAinfo['TOA']
    TOA_t_num=TOAinfo['TOA_t_num']
    ind_keep, phoneIDs= zip(*[(i,p) for i,p in enumerate(TOAinfo['phoneIDs'].astype(int)) if p in phoneIDs_all])
    TOA = TOA[:,ind_keep]
    
    ind_phone = np.array([phoneIDs_all.index(p) for p in phoneIDs])
    phoneLoc = phoneLoc_all[ind_phone]

    
    #TOA[:,0]=-1 # remove one phone
    # remove bad receivers
    nvalid = np.sum(TOA>1e-8,axis=0)
    ind=nvalid<5    
    TOA[:,ind] = -1  
    # 1. remove signals with receiver number smaller than $ignorePhoneNum$
    # 2. remove signals with less than two adjacent well posed signals 
    nvalid = np.sum(TOA>1e-8,axis=1)
    #r=np.random.rand(len(nvalid))
    ind=(nvalid>ignorePhoneNum)  #&  (r>0.5)
    neigh_valid_sum = np.convolve(nvalid>=minPhoneNum, np.ones(5), mode='same')
    ind_neigh = neigh_valid_sum>=0  # do nothing
    ind = ind & ind_neigh
    #ind = ind & (TOA[:,-1]>1e-8) # the phone 6 must be detected, only for Cabot 
    TOA= TOA[ind]
    TOA_t_num=TOA_t_num[ind]
    mask   = TOA>1E-8;
    
    
    # drop head and tail signals with 3 or fewer receivers
    nvalid=np.sum(mask, axis=1)    
    if dam == 'McCloud9999':
        minPhoneNum=2 #only for MacCloud
    ind=np.where(nvalid>=minPhoneNum)[0] 
    if len(ind)<=2:
        return None
    # only for tail tracking, at most track 5000 points
    length = ind[-1]-ind+1
    ind_ind=np.where(length<numTrack)[0][0]
    ind = ind[ind_ind:]    
    if len(ind)<=2:
        return None

    i1,i2=ind[0],ind[-1]+1
    TOA=TOA[i1:i2]
    TOA_t_num=TOA_t_num[i1:i2]
    mask=mask[i1:i2]     
    # drop columns
    nvalid= np.sum(mask, axis=0)
    ind   = np.where(nvalid<=0)[0]
    TOA[:,ind]=-1000
    mask=TOA>1e-3
    # phoneLoc=phoneLoc[ind,:]    
    TOA[~mask] = TOA.max()+999999
    time = np.min(TOA, axis=1)[:,None]
    TOA[~mask]=-1
    
    #clip idle time interval
    dt = np.diff(time[:,0])
    tol_window = 600
    dt[dt>tol_window] = tol_window
    time_new = np.insert(np.cumsum(dt), 0, 0.)[:,None]
    TOA_new  = TOA+(time_new-time)
    TOA_new[~mask]=-1
    
    #time= ( np.sum(TOA*mask, axis=1)/np.sum(mask,axis=1) )[:,None]
    data=(time_new, TOA_new, phoneLoc, np.array(phoneIDs), mask,TOA_t_num)
    return data


def get_PRI(toa_noisy, mask):
    PRI= np.concatenate( [ np.diff(toa_noisy[mask[:,i],i]) for i in range(toa_noisy.shape[1])] )
    PRI= np.percentile(PRI, 20)
    return PRI



## physics-based network model   
class Model(torch.nn.Module):
    def __init__(self, fileSystem, data, TagID,dam='LMN',
                 temperature_ref=15, dis_detect=100,
                 region_lb=[0,0,0], region_ub=[1,1,1],
                 ignorePhoneNum=1,Vel_ref = 5.0,nSplit=8000,
                 layers=[200]+[50]*6+[3],ignoreZ=False,
                 sdfNetFile=None, initNet=True,loc_sys=None,
                 distance_threshold=3.0,
                 std_deviation_factor=5.0,
                 time_shift_factor=0.3,
                 uq_std=1.0,
                 ):
        '''        

        Parameters
        ----------
        fileSystem : class
            file name system.
        data : tuple
            data generated from getTrainData.
        TagID : int
            Tag index.
        temperature_ref : float, optional
            reference temperature. The default is 15.
        dis_detect : float, optional
            maximum detection distance of hydrophone. The default is 100.
        region_lb, region_ub : np.ndarray, optional
            lower and upper box bound osf the region     
        ignorePhoneNum : int, optional
            a siginal with $ignorePhoneNum$ or fewer receivers will be ignored.
            The default is 1.
        pRM : float, optional
            maximum ratio of TOA point to be removed.
            The default is 0.05.            
        noise_toa_ref : float, optional
            reference standard derivation of toa noise (0 mean is assumed).
            The default is 0.05.
        noise_soundSpeed_ref : float, optional
            reference standard derivation of soundSpeed noise (0 mean is assumed).
            The default is 0.01.
        layers : list, optional
            network architecture. The default is [200]+[50]*6+[3].            
        Vel_ref : float, optional
            used to control the frequency of network. The default is 5.
        ignoreZ : bool, optional
            the third dimension is ignored if ignoreZ is True. The default is 5.            
        Returns
        -------
        None.

        '''        
        super(Model, self).__init__()
        self.loc_sys=loc_sys
        self.FNS=fileSystem
        self.TagID=TagID
        self.dam = dam
        self.layers = layers
        self.dis_detect =dis_detect
        self.ignorePhoneNum=ignorePhoneNum
        self.dim = 3 if not ignoreZ else 2
        self.Vel_ref = Vel_ref
        self.distance_threshold = distance_threshold
        self.std_deviation_factor = std_deviation_factor
        self.time_shift_factor = time_shift_factor
        self.uq_std = uq_std
         
        # split time domain
        #tol_window = 600
        #mask = data[-2] 
        #valid = np.sum(mask, axis=1)>=self.dim+1
        #nValid = np.sum(valid)
        #nDomain=self.nDomain = int(np.ceil(nValid/nSplit))
        #nSplit_real = mask.shape[0]/nDomain
        #logger.info("here", nDomain, nValid, nSplit)
        #time,TOA = data[:2]
        #for i in range(1,nDomain):
        #    ind = int(i*nSplit_real)
        #    time[ind:] += tol_window
        #    TOA[ind:]  += tol_window
        #data = (time, TOA, *data[2:])

        
        # load data
        self.phoneIDs = data[-3]
        self.data = [ Numpy2Tensor(d) for d in data] 
        self.temperature_ref=temperature_ref
        self.soundSpeed_ref = soundSpeed_freshWater(temperature_ref)
        self.region_lb=Numpy2Tensor(region_lb).reshape(1,3)
        self.region_ub=Numpy2Tensor(region_ub).reshape(1,3)
        self.sdfFun = lambda x: x[:,0:1]*0-1
        if sdfNetFile is not None:
            if sdfNetFile.lower()=='downstreamdam':
                self.sdfFun = lambda x: x[:,0:1] -0.3
                self.region_ub[0,0] = zero
            elif sdfNetFile.lower()=='upstreamdam':
                self.sdfFun = lambda x: -x[:,0:1]+0.3
                self.region_lb[0,0] = zero
            else:
                logger.error(f'unknow para sdfNetFile={sdfNetFile}')
        toa_noisy = self.data[1]
        self.weight=torch.ones_like(toa_noisy)
        self.useGivenWeight = False
        self.toa_noise = torch.zeros_like(toa_noisy)
        phoneLoc = self.data[2]
        self.phoneLoc_noise = torch.zeros_like(phoneLoc)        
        if initNet:
            self.initNet()            
        return
    def initNet(self):
        '''
        define the network and 
        other training parameters
        ----------
        top : time of ping for each signal
        ss  : sound speed 
        toa_tau : variance of toa for each phone
        ss_tau  : variance of sound speed
        ss_mean : avergae of sound speed
        '''
        t,toa_noisy=self.data[:2]
        Nhp=toa_noisy.shape[1]        
        Nt=t.shape[0]
        #top=torch.nn.Parameter(torch.zeros(Nt,1))
        top=torch.nn.Parameter(torch.randn(Nt,1)*0.2)
        toa_tau  = torch.nn.Parameter(torch.zeros(1,Nhp))  
        stat = [toa_tau,]
        sol = [top, ]        
        self.paras = [sol , stat]
        self.net=MFNN( self.layers ).to(torch.defaultDevice).to(dtype=torch.defaultReal)
        self.net.init_params()
        
        self.features = self.Vel_ref*torch.randn(self.layers[0]//2,1)
        self.tmin, self.tmax=t.min(),t.max()
        
        n_ctrl_pts = int(len(t)/500) + 1
        logger.info(f"{n_ctrl_pts} control points are generated")
        if not self.useGivenWeight:
            self.net_tau = BSplineModel3D(n_ctrl_pts=n_ctrl_pts, dims=Nhp, k=1)
            t_norm= (t-self.tmin)/(self.tmax-self.tmin)
            self.basisMatrix_tau =self.net_tau.build_basis_matrix(t_norm)      
        if not self.useGivenWeight:
            self.weight=torch.ones_like(toa_noisy)
        return
    
    def initOpt(self, lr=1e-3, lr_lambda=lambda step:1, weight_decay=1e-4, ):
        '''        
        initialize optimizers
        Parameters
        ----------
        lr : float, optional
            learning rate. The default is 1e-3.
        lr_lambda : function, optional
            updating rule of learning rate. The default is lambda step:1.
        weight_decay : float, optional
            L2 normailization weight. The default is 1e-4.

        '''
        # define optimizer
        net_tau_params = [] if self.useGivenWeight else list(self.net_tau.parameters())
        self.opt_Adam    = torch.optim.Adam(self.paras[0] + self.paras[1] +list(self.net.parameters())+net_tau_params, 
                                            lr=lr, weight_decay=weight_decay, eps=1e-14)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR( self.opt_Adam, lr_lambda=lr_lambda )   
        self.opt_BFGS = torch.optim.LBFGS(
            self.paras[0]  +list(self.net.parameters()),
            lr=1.0,
            max_iter=20,
            max_eval=25,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            history_size=100,
            line_search_fn='strong_wolfe',  # enables proper curvature-guided step sizes
        )     
    
  

        
    def lossVec0(self, paras, data):
        "define loss matrix for assessment"
        xyz, top, stat  = self.decodeInput(paras, data)
        t, toa_noisy, phoneLoc, phoneIDs, mask, TOA_t_num=data
        mask = mask.to(torch.bool)  
        # position constriant
        dis = xyz[:,None,:self.dim]-phoneLoc[None,:,:self.dim]
        r  = torch.norm(dis, dim=2)
        noise_toa = top+r/self.soundSpeed_ref-toa_noisy
        loss_pos = (self.weight**2*noise_toa.square())[mask].sum()/(self.weight**2)[mask].sum()        
        # delta = 3/self.soundSpeed_ref
        # loss_pos=self.getHubLoss( noise_toa, delta)
        # loss_pos = (loss_pos)[mask].mean()        
        return loss_pos,
    
    def getHubLoss(self, noise_toa, delta):
        noise_toa = torch.abs(noise_toa)
        mask_small = noise_toa<=delta
        loss_pos_small  = 0.5*noise_toa**2 #)/2*(1/3)**2).mean()
        loss_pos_large  = delta*(noise_toa-0.5*delta) #0.5*(torch.square(xyz_LF[:, :self.dim]-xyz_LF_NN[:, :self.dim])/2*(1/3)**2).mean()
        loss_pos = loss_pos_small*mask_small + (~mask_small)*loss_pos_large                        
        return loss_pos
    
    def lossVec(self, paras, data, BFGS=False):
        "define loss matrix for traning"
        xyz, top,  stat = self.decodeInput(paras, data)
        toa_tau,  =stat     
        t, toa_noisy, phoneLoc, phoneIDs, mask, TOA_t_num=data
        mask = mask.to(torch.bool)
        # position constriant
        # add small perturbation to phone location to avoid saddle points
        phoneLoc0 = phoneLoc + (torch.randn(phoneLoc.shape)*0.2 if not BFGS else 0) + self.phoneLoc_noise
        dis = xyz[:,None,:self.dim]-phoneLoc0[None,:,:self.dim]
        r  = torch.norm(dis, dim=2)
        noise_toa = top+r/self.soundSpeed_ref-toa_noisy + self.toa_noise
        if not BFGS and not self.useGivenWeight:
            with torch.no_grad():
                dt= self.distance_threshold/self.soundSpeed_ref
                noise_ratio = torch.minimum( torch.abs(noise_toa)//(dt), 10*one )
                weight = 1/noise_ratio*toa_tau/torch.max(toa_tau)*torch.ones_like(noise_toa)
                #weight = toa_tau/torch.max(toa_tau)*torch.ones_like(noise_toa)
                weightMask = (torch.abs(noise_toa)>dt)&mask                
                weight[~weightMask] = 1
                validNum_min = 0 
                validNum = torch.sum(mask,axis=1)
                valid = validNum>=validNum_min
                weight[valid] *=  validNum[valid][:,None]**2 # 2**(validNum[valid][:,None]-validNum_min+1)
                beta = 0.996
                #weight[mask] /= weight[mask].mean()+1e-10
                self.weight = beta*self.weight + (1-beta)*weight                
        # delta = 3/self.soundSpeed_ref
        # loss_pos=self.getHubLoss( noise_toa, delta)
        # loss_pos = (loss_pos*toa_tau**2 - torch.log(toa_tau))[mask].mean()        
        
        loss_pos = (self.weight**2*torch.square(noise_toa)/2*toa_tau**2 - torch.log(toa_tau))[mask].mean()        
        with torch.no_grad():
            coeff  = torch.abs(toa_tau).square().mean()
        sdf=torch.relu(self.sdfFun(xyz[:,:2]))/self.soundSpeed_ref
        loss_pos += coeff*sdf.square().sum()/(torch.sum(sdf>0)+1)
        return loss_pos,

    def loss_func(self, paras, data, BFGS=False, weight_decay=0):
        "define loss fuction"
        if not BFGS:
            self.opt_Adam.zero_grad()
        else:
            self.opt_BFGS.zero_grad()        
        lossVec = self.lossVec(paras, data, BFGS=BFGS)
        loss_all=torch.cat( [ k.flatten() for k in lossVec ] )
        loss = loss_all.sum()
        for name,params in self.net.named_parameters():
            if 'weight' in name:
                loss += params.square().sum()*weight_decay
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.paras[:-1] + self.paras[-1], clip_value=1E6)
        return loss

    def net_forward_feature(self, t):
        "forward propagation of network model"
        t= (t-(self.tmax+self.tmin)/2)/(self.tmax-self.tmin)*2
        phi = torch.matmul(t, self.features.T)
        phi = torch.cat((phi, phi-0.5), dim=1)
        phi *=torch.pi
        t   = torch.cos(phi)        
        xyz = self.net(t)
        
        fun2 = lambda x,n: 1+torch.exp(-self.std_deviation_factor*x/n)
        tau  = self.basisMatrix_tau @ self.net_tau.ctrl_pts
        tau  = fun2(tau,1)         
        return xyz, tau

    def fixedNoiseSampling(self, paras, data, std_dis=1.0, std_phoneLoc=0.5):
        t, toa_noisy, phoneLoc, phoneIDs, mask, TOA_t_num= data
        self.toa_noise = std_dis/self.soundSpeed_ref*torch.minimum(torch.maximum(torch.randn_like(toa_noisy), -3*one), 3*one)
        self.phoneLoc_noise = std_phoneLoc*torch.randn_like(phoneLoc)
        return
    def fixedDistanceSampling(self, xyz, paras, data, std_dis=1.0, std_phoneLoc=0.0):
        t, toa_noisy, phoneLoc, phoneIDs, mask, TOA_t_num= data
        dis = xyz[:,None,:self.dim]-phoneLoc[None,:,:self.dim]
        r  = torch.norm(dis, dim=2)
        xyzNew= xyz + std_dis*torch.minimum(torch.maximum(torch.randn_like(xyz), -3*one), 3*one)
        dis = xyzNew[:,None,:self.dim]-phoneLoc[None,:,:self.dim]
        rNew  = torch.norm(dis, dim=2)  
        
        self.toa_noise =(rNew-r)/self.soundSpeed_ref
        self.phoneLoc_noise = std_phoneLoc*torch.randn_like(phoneLoc)
        return    
    def decodeInput(self, paras, data):
        "decode network training parameters"
        sol, stat=paras
        top,=sol
        toa_tau,  = stat
        t, toa_noisy, phoneLoc, phoneIDs, mask, TOA_t_num= data
        xyz,toa_tau = self.net_forward_feature(t)
        fun = lambda x: torch.tanh(x)  #lambda x: torch.clamp(x,-1,1)
        scale = torch.tensor([[1., 1., 0.01]])
        xyz = fun(xyz*scale)*(self.region_ub-self.region_lb)/2 +  (self.region_ub+self.region_lb)/2
        
        #top = t-self.dis_detect/self.soundSpeed_ref*fun(top)
        top = t-self.time_shift_factor*fun(top)
        #fun2 = lambda x,n: 1+torch.exp(-10*x/n)
        #toa_tau  = fun2(toa_tau,1)  
        stat = [toa_tau,]
        return xyz, top,  stat    
        
    def train(self, nIterAdam, nIterBFGS,
              lr, lr_lambda, weight_decay,
              displayAdam=100, displayBFGS=10, 
              nSamples=20, 
              sampleMethod=['no noise', 'fixed noise', 'fixed distance'][1],
              saveCSV=None):
        assert sampleMethod.lower() in ['no noise', 'fixed noise', 'fixed distance']
        xyz_all = []
        paras_decoded0=None
        for i in range(nSamples):
            logger.info(f"==========training for {i}th case========")
            if i==0: # no noise                      
                self.useGivenWeight = False
            else: 
                self.useGivenWeight = False
                if sampleMethod.lower()=='no noise': pass
                if sampleMethod.lower()=='fixed noise':
                    self.fixedNoiseSampling(self.paras, self.data)
                if sampleMethod.lower()=='fixed distance':
                    xyz = paras_decoded0[0]
                    self.fixedDistanceSampling(xyz, self.paras, self.data,
                                               std_dis=self.uq_std)
            paras_decoded = self.trainSingle(nIterAdam, 
                                   nIterBFGS=nIterBFGS,
                                  lr=lr, lr_lambda=lr_lambda, weight_decay=weight_decay,
                                  displayAdam=displayAdam, displayBFGS=displayBFGS,)            
            xyz = paras_decoded[0]
            xyz_all.append(xyz)
            if i==0:
                paras_decoded0 = paras_decoded
        xyz_all = torch.stack(xyz_all, dim=0)
        paras_decoded0 =(xyz_all, *paras_decoded0[1:])
        track_NN =self.getTrack(paras_decoded0, saveCSV=saveCSV)
        return track_NN
                
    def getSolution(self):
        with torch.no_grad():
            paras_decoded=self.decodeInput(self.paras, self.data)
        return paras_decoded        
    @timing
    def trainSingle(self, nIterAdam, nIterBFGS,
              lr, lr_lambda, weight_decay,
              displayAdam=100, displayBFGS=10,
              patience=15):
        '''
        Parameters
        ----------
        nIterAdam : int
            Number of Adam iterations.
        nIterBFGS : int
            Number of full-batch L-BFGS iterations run after Adam.
            L-BFGS uses second-order curvature and strong Wolfe line search
            so it converges the residual much faster than Adam alone.
            Recommended: 100-300. 0 disables.
        lr : float
            Adam learning rate.
        lr_lambda : function
            LR schedule for Adam.
        weight_decay : float
            L2 regularisation weight.
        displayAdam : int, optional
            Log status every this many Adam epochs. Default 100.
        displayBFGS : int, optional
            Log status every this many L-BFGS epochs. Default 10.
        patience : int, optional
            Adam early-stop patience in units of displayAdam epochs (i.e.
            patience * displayAdam Adam steps without improvement triggers
            early stop). Default 15 (= 1500 Adam steps).
        Returns
        -------
        paras_decoded : tuple
        '''
        self.initNet()
        self.initOpt(lr=lr, lr_lambda=lr_lambda, weight_decay=0)

        error_RMSE_min, loss_likelihood_min = 1e10, 1e10
        no_improve_count = 0
        paras_decoded = self.getSolution()  # ensure always defined even if nIterAdam=0
        for epoch in range(nIterAdam):
            self.loss_func(self.paras, self.data,
                           BFGS=False,
                           weight_decay=weight_decay)
            self.opt_Adam.step()
            self.lr_scheduler.step()
            if epoch == 0 or (epoch + 1) % displayAdam == 0:
                error_RMSE, loss_likelihood = self.record_status('Adam', epoch)
                improved = (error_RMSE < error_RMSE_min * 0.9995
                            or loss_likelihood < loss_likelihood_min * 0.9995)
                if improved:
                    error_RMSE_min = min(error_RMSE.item(), error_RMSE_min)
                    loss_likelihood_min = min(loss_likelihood.item(), loss_likelihood_min)
                    paras_decoded = self.getSolution()
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
        # Full-batch L-BFGS polishing with strong Wolfe line search.
        # ----------------------------------------------------------------
        if nIterBFGS > 0:
            for epoch in range(nIterBFGS):
                self.opt_BFGS.step(
                    lambda: self.loss_func(
                        self.paras, self.data, BFGS=True, weight_decay=weight_decay))
                if epoch == 0 or (epoch + 1) % displayBFGS == 0:
                    error_RMSE, loss_likelihood = self.record_status('BFGS', epoch)
                    if (error_RMSE.item() < error_RMSE_min * 0.9995
                            or loss_likelihood.item() < loss_likelihood_min * 0.9995):
                        error_RMSE_min = min(error_RMSE.item(), error_RMSE_min)
                        loss_likelihood_min = min(loss_likelihood.item(), loss_likelihood_min)
                        paras_decoded = self.getSolution()
        self.plot_train_history()
        return paras_decoded
    def record_status(self,optimizer, epoch, ):
        error_RMSE,     = self.lossVec0(self.paras, self.data)
        loss_likelihood, = self.lossVec(self.paras, self.data)
        
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
        paras=self.decodeInput(self.paras, self.data)        
        #stat = paras[-1]
        #logger.debug(f"{torch.cat([d.mean(axis=0).flatten() for d in stat])}")
        
        #self.saveParas(self.TagID)
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
    
    
    def getTrack(self, paras_decoded,  saveCSV=None):    
        data = self.data
        xyz_all=paras_decoded[0]
        xyz = torch.mean(xyz_all, dim=0)
        xyz_std = torch.std(xyz_all, dim=0, unbiased=False)        
        top=paras_decoded[-2]
        toa_tau,  =paras_decoded[-1]
        t, toa_noisy, phoneLoc, phoneIDs, mask, TOA_t_num=data[:6]
        TOA_t_num=Tensor2Numpy(TOA_t_num)
        timestamps = pd.to_datetime(TOA_t_num, unit='D').round('s')
        mask= Tensor2Numpy(mask)
        mask=np.bool_(mask)
        toa_noisy=Tensor2Numpy(toa_noisy)
        dt=np.diff(TOA_t_num)*24*3600
        t_total = np.sum( dt[dt<300] )      
        PRI = get_PRI(toa_noisy, mask)
        efficiency=xyz.shape[0] / (t_total/PRI)
        logger.info(f"PRI= {PRI}, {efficiency,xyz.shape[0],t_total}")   
        nValid = mask.sum(axis=1)
        track_NN={'t_datetime':timestamps,
                't_num': np.array([ ti.timestamp() for ti in timestamps]),
                'XYZ': Tensor2Numpy(xyz),
                'XYZ_std': Tensor2Numpy(xyz_std),
                'phoneLoc':Tensor2Numpy(phoneLoc),
                'XYZ_all':Tensor2Numpy(xyz_all),
                'phoneIDs':list(Tensor2Numpy(phoneIDs).astype(int)),
                'toa_noisy':toa_noisy,
                'top': Tensor2Numpy(top),
                'weight':Tensor2Numpy(self.weight),
                'toa_tau':Tensor2Numpy(toa_tau),
                'soundSpeed':self.soundSpeed_ref,
                'efficiency':efficiency,
                'nValid': nValid,
                'PRI':PRI,
                'dim':self.dim,
                }        
        track_NN = self.XYZ2ENE(self.loc_sys, track_NN)
        #self.writeTrack(track_NN, saveCSV=saveCSV)    
        return track_NN
    @classmethod    
    def writeTrack(cls, track_NN,saveCSV=None):
        error = np.zeros_like(track_NN['XYZ'])*np.nan
        if 'XYZE' in track_NN:
            error = track_NN['XYZ']- track_NN['XYZE']            
        if "XYZ_std" not in track_NN:
            track_NN['XYZ_std'] = np.zeros_like(track_NN['XYZ'])
        df = pd.DataFrame({'datetime':track_NN['t_datetime'],
                           'X':track_NN['XYZ'][:,0],
                           'Y':track_NN['XYZ'][:,1],
                           'Z':track_NN['XYZ'][:,2],
                           'X std':track_NN['XYZ_std'][:,0],
                           'Y std':track_NN['XYZ_std'][:,1],
                           'Z std':track_NN['XYZ_std'][:,2],                           
                           'X error':error[:,0],
                           'Y error':error[:,1],
                           'Z error':error[:,2],                           
                           'nValid':track_NN['nValid']})
        phoneIDs = track_NN['phoneIDs']
        if 'top' in  track_NN:
            df['top']=track_NN['top'] 
        for i, phoneID in enumerate(phoneIDs):
            df[f'phone {phoneID}'] = track_NN['toa_noisy'][:,i]
        if 'weight' in track_NN:
            for i, phoneID in enumerate(phoneIDs):
                df[f'weight {phoneID}'] = track_NN['weight'][:,i]
            for i, phoneID in enumerate(phoneIDs):
                df[f'tau {phoneID}'] = track_NN['toa_tau'][:,i]
        
        df [['phone X', 'phone Y', 'phone Z']] = np.nan 
        if len(phoneIDs)>len(df):
            empty_rows = pd.concat([pd.DataFrame([[None]*df.shape[1]], columns=df.columns)] * (len(phoneIDs)-len(df)), ignore_index=True)
            df = pd.concat([df, empty_rows], ignore_index=True)
        df.loc[df.index[:len(phoneIDs)], ['phone X', 'phone Y', 'phone Z']] = track_NN['phoneLoc']
        if 'PRI' in track_NN:
            df['PRI']   =  track_NN['PRI']  
        if saveCSV:
            df.to_csv(saveCSV,index=False)   
            filePickle = saveCSV + '.pickle'
            with open(filePickle, 'wb') as f:
                pickle.dump(track_NN, f)
            logger.info(f"track saved to {saveCSV}")
        return 
    @classmethod
    def readTrack(cls, saveCSV=None):
        assert saveCSV is not None
        filePickle = saveCSV + '.pickle'
        with open(filePickle, 'rb') as f:
            track_NN = pickle.load(f)
        return track_NN
    
    @classmethod
    def trimTrack(cls, trackNN, datetime_start, datetime_end):
        t_datetime=trackNN['t_datetime']
        ind = (datetime_start<=t_datetime) & (t_datetime<= datetime_end) 
        trimed_trackNN={'t_datetime':trackNN['t_datetime'][ind],
                't_num': trackNN['t_num'][ind],
                'XYZ': trackNN['XYZ'][ind],
                'XYZ_std':trackNN['XYZ_std'][ind],
                'XYA_all':trackNN['XYZ_all'][:,ind],
                'phoneIDs':trackNN['phoneIDs'],
                'phoneLoc':trackNN['phoneLoc'],
                'toa_noisy':trackNN['toa_noisy'][ind],
                'weight':trackNN['weight'][ind],
                'top':trackNN['top'][ind],
                'toa_tau':trackNN['toa_tau'][ind],
                'nValid':trackNN['nValid'][ind],
                'soundSpeed':trackNN['soundSpeed'],
                'efficiency':trackNN['efficiency'],
                'PRI':trackNN['PRI'],
                'dim':trackNN['dim'],
                }  
        try:       
            trimed_trackNN['efficiency']=(trackNN['efficiency']*
                                        trimed_trackNN['t_num'].shape[0]/(trackNN['t_num'].shape[0]+1E-10)
                                        *(trackNN['t_num'].max()-trackNN['t_num'].min())/(trimed_trackNN['t_num'].max()-trimed_trackNN['t_num'].min()+1E-10)
                                        )
        except:
            trimed_trackNN['efficiency']=0
        return trimed_trackNN
    @classmethod
    def trackCat(cls,trackA, trackB):
        assert len(trackB)>0
        if len(trackA)==0:
            return trackB
        tmidA = trackA['t_num'].mean()
        tmidB = trackB['t_num'].mean()
        new_track =dict()
        new_track['soundSpeed']=trackB['soundSpeed']
        new_track['phoneLoc']=trackB['phoneLoc']
        new_track['phoneIDs']=trackB['phoneIDs']
        new_track['PRI']=trackB['PRI']
        new_track['dim']=trackB['dim']
        
        for key in list(trackA.keys()):
            if key not in ['soundSpeed','efficiency','phoneLoc','dim','phoneIDs','PRI']:
                dim=1 if key in ['XYZ_all'] else 0
                if tmidA<tmidB:
                    new_track[key]=np.concatenate((trackA[key], trackB[key]), axis=dim)
                else:
                    new_track[key]=np.concatenate((trackB[key], trackA[key]), axis=dim)
        new_track['efficiency'] = ( (trackA['XYZ'].shape[0]+trackB['XYZ'].shape[0])/
                             (trackA['XYZ'].shape[0]/trackA['efficiency']
                             +trackB['XYZ'].shape[0]/trackB['efficiency'])
                             )
        return new_track
    
    @classmethod
    def XYZ2ENE(cls, loc_sys, track):
        import copy
        trackNew = copy.deepcopy(track)
        phoneInfo =loc_sys.phoneInfo
        ENE0, theta=phoneInfo['ENE0'], phoneInfo['theta']
        trackNew['phoneLoc'] = loc_sys.XYZ2ENE(track['phoneLoc'], ENE0, theta)
        trackNew['XYZ'] = loc_sys.XYZ2ENE(track['XYZ'], ENE0, theta)
        cos =np.cos(theta); sin =np.sin(theta)
        if "XYZ_std" in trackNew:
            xyz_std=track['XYZ_std']
            ene_std=xyz_std+0
            ene_std[:,0] = np.sqrt(xyz_std[:,0]**2*cos**2+xyz_std[:,1]**2*sin**2)
            ene_std[:,1] = np.sqrt(xyz_std[:,0]**2*sin**2+xyz_std[:,1]**2*cos**2)
            trackNew['XYZ_std']=ene_std
        if 'XYZ_all' in trackNew:
            trackNew['XYZ_all'] = loc_sys.XYZ2ENE(track['XYZ_all'], ENE0, theta)
        return trackNew

