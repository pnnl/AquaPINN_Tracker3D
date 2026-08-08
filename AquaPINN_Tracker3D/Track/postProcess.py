# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:26:37 2024

@author: linx882
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from .ML_NN import Model
import matplotlib.gridspec as gridspec
plt.rcParams.update({'figure.max_open_warning': 0})
from ..SharedLibs.decorators import timing
from scipy.interpolate import interp1d


def boxPlot(ax, err, ylim):
    ind=np.isfinite(err)
    err=err[ind]
    flierprops = dict(marker='+', markerfacecolor='red', markeredgecolor='red')
    ax.boxplot(err,flierprops=flierprops, vert=True)
    mean = np.nanmean(err)
    median  = np.nanmedian(err)
    rmse    = np.sqrt(np.nanmean(err**2))
    ax.axhline(mean, color='k', linestyle='--')
    ax.set_ylim(ylim)
    ax.axis('off')
    label=f'Mean: {mean:.2f}   Median: {median:.2f}  RMSE: {rmse:.2f}'
    return label

def trackInterpolate(t_num_target,track_GPS):
    ind = (t_num_target<track_GPS['t_num'].max()) & (t_num_target>track_GPS['t_num'].min())
    t_num=t_num_target[ind]
    ind_GPS = (t_num.max()+5>track_GPS['t_num']) & (track_GPS['t_num']>t_num.min()-5)
    xyzG= track_GPS['XYZ'][ind_GPS]
    tG_num = track_GPS['t_num'][ind_GPS]   
    if len(tG_num)<3:
        return None, None, None
    f_cubic = interp1d(tG_num, xyzG, axis=0,kind='cubic',
                       bounds_error=False, fill_value=np.nan)
    xyz = f_cubic(t_num)
    return ind,xyz, ind_GPS
def addGPS2Track(track_NN,track_GPS):
    if track_GPS is None:
        return track_NN
    t_num_target = track_NN['t_num']
    ind,xyz, ind_GPS = trackInterpolate(t_num_target,track_GPS)
    if ind is None:
        return track_NN
    track_NN['XYZE'] = np.zeros_like(track_NN['XYZ'])*np.nan
    track_NN['XYZE'][ind] = xyz
    return track_NN

def getTOA(track_NN, mode='residual'):
    xyz=track_NN['XYZ']
    phoneLoc=track_NN['phoneLoc']
    toa_noisy = track_NN['toa_noisy']
    if 'top' not in track_NN:
        return None
    top       = track_NN['top']
    soundSpeed=track_NN['soundSpeed']
    dim = track_NN['dim']
    dis = xyz[:,None,:dim]-phoneLoc[None,:,:dim]
    r  = np.linalg.norm(dis, axis=2)    
    toa_noisy[toa_noisy<1e-3] = np.nan
    assert mode in ['residual', 'toa', 'track']
    if mode == 'residual':
        err_toa = top+r/soundSpeed-toa_noisy
    elif mode == 'toa':
        err_toa = toa_noisy
    elif mode == 'track':
        err_toa = r/soundSpeed     
    return err_toa

def getTDOA(track_NN, mode='residual'):
    xyz=track_NN['XYZ']
    phoneLoc=track_NN['phoneLoc']
    toa_noisy = track_NN['toa_noisy']
    soundSpeed=track_NN['soundSpeed']
    dim = track_NN['dim']
    dis = xyz[:,None,:dim]-phoneLoc[None,:,:dim]
    r  = np.linalg.norm(dis, axis=2)    
    toa_noisy[toa_noisy<1e-3] = np.nan
    assert mode in ['residual', 'toa', 'track']
    if mode == 'residual':
        err_toa = 0+r/soundSpeed-toa_noisy
    elif mode == 'toa':
        err_toa = toa_noisy
    elif mode == 'track':
        err_toa = r/soundSpeed
    err_tdoa= err_toa[:,None,:] - err_toa[:,:,None]         
    return err_tdoa

colors=['r','g','b','k','y']
def plotTOA(t_datetime, err_toas,phoneIDs, labels,
             baseName, outputdir):
    
    for i,phoneIDi in enumerate(phoneIDs):
        plt.figure()
        for k,err_toa in enumerate(err_toas):
            ind = np.isfinite(err_toa[:,i])
            if np.sum(ind)<=0:
                continue            
            plt.plot(t_datetime[ind], err_toa[ind,i], 'o', markersize=2, label=labels[k],color=colors[k])
        plt.tick_params(axis='x', labelrotation=30)
        plt.legend(loc='best')
        file = f"{baseName}_toa{phoneIDi}.jpg"      
        
        plt.savefig(os.path.join(outputdir,file),
                    dpi=300, bbox_inches='tight', facecolor='white')            
        plt.close()
    return
def plotTDOA(t_datetime, err_tdoas,phoneIDs, labels,
             baseName, outputdir):
    for i,phoneIDi in enumerate(phoneIDs):
        for j,phoneIDj in enumerate(phoneIDs):  
            if i>=j:
                continue
            plt.figure()
            for k,err_tdoa in enumerate(err_tdoas):
                ind = np.isfinite(err_tdoa[:,i,j])
                if np.sum(ind)<=0:
                    continue            
                plt.plot(t_datetime[ind], err_tdoa[ind,i,j], 'o', markersize=2, label=labels[k],color=colors[k])
            plt.tick_params(axis='x', labelrotation=30)
            plt.legend(loc='best')
            file = f"{baseName}_Pair{phoneIDi}-{phoneIDj}.jpg"      
            
            plt.savefig(os.path.join(outputdir,file),
                        dpi=300, bbox_inches='tight', facecolor='white')            
            plt.close()
    return

def addTrajectoryErrorband(ax, x,y,x_std,y_std, color='b',alpha=0.2):
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1E-6
    nx = -dy / norm
    ny = dx / norm

    x_upper = x + x_std * nx
    y_upper = y + y_std * ny
    x_lower = x - x_std * nx
    y_lower = y - y_std * ny

    x_band = np.concatenate([x_upper, x_lower[::-1]])
    y_band = np.concatenate([y_upper, y_lower[::-1]])

    ax.fill(x_band, y_band, color=color, alpha=alpha, edgecolor='none')
    return ax     

    
@timing
def plot_track_OneFigureEqual(track_NN, track_GPS=None, figTitle=None,
                         fout=None, testInfo=None,close=False,description="", method="NN"):    
    phoneIDs = track_NN['phoneIDs']
    xyz_base = track_NN['phoneLoc'][:1]    
    phoneLoc = track_NN['phoneLoc']-xyz_base
    xyz =track_NN['XYZ']-xyz_base
    if "XYZ_std" in track_NN:
        xyz_std = track_NN['XYZ_std']
    else:
        xyz_std = np.zeros_like(track_NN['XYZ'])
    t_num=track_NN['t_num']
    t_datetime=track_NN['t_datetime']
    if len(t_num)<3:
        return

    if track_GPS is not None:        
        ind, xyzE, ind_GPS = trackInterpolate(t_num,track_GPS)
        xyzE -= xyz_base
        if ind is None:
            track_GPS = None
        else:            
            t_num = t_num[ind]
            t_datetime=t_datetime[ind]
            xyz = xyz[ind]
            xyz_std= xyz_std[ind]
            xyzG= track_GPS['XYZ'][ind_GPS]-xyz_base
            tG_datetime=  track_GPS['t_datetime'][ind_GPS]     
    
    fig = plt.figure(figsize=(21, 8))
    nc=1
    gs = gridspec.GridSpec(2,4*nc, figure=fig, wspace=0.2, hspace=0.4)
    
    
    axU0 = fig.add_subplot(gs[0,0])  # 
    if track_GPS is not None:
        axU0.plot(xyzG[:,0],xyzG[:,1],'r-', label='GPS')
    labelNN=f"{method},{track_NN['efficiency']*100:0.1f}%"
    axU0.plot(xyz[:,0],xyz[:,1],'b-o',label=labelNN,markersize=1)
    addTrajectoryErrorband(axU0,xyz[:,0],xyz[:,1],xyz_std[:,0], xyz_std[:,1], color='b', alpha=0.3)
    axU0.plot(phoneLoc[:,0],phoneLoc[:,1],'rp',markersize=10, label='phones')
    for i in range(len(phoneLoc)):
        axU0.text(phoneLoc[i,0],phoneLoc[i,1]-1, f"{phoneIDs[i]}",
                  ha='center', va='bottom', fontsize=8)        
    axU0.set_xlabel('x')
    axU0.set_ylabel('y')
    axU0.legend(loc='best')
    axU0.set_title("Tracking number:%d, efficiency:%.2f%%"%(xyz.shape[0], track_NN['efficiency']*100))
    
    for kk in range(3):
        flag=['x','y', 'z'][kk]
        axU1 = fig.add_subplot(gs[0,kk+1])  # 
        if track_GPS is not None and not (track_GPS['GPS_Z_valid']==False and kk==3):
            axU1.plot(tG_datetime,xyzG[:,kk],'r-', label='GPS')            
        axU1.plot(t_datetime,xyz[:,kk],'b-o',markersize=1)
        plt.fill_between(t_datetime, 
                         (xyz[:,kk] - 2 * xyz_std[:,kk]).flatten(), 
                         (xyz[:,kk] + 2 * xyz_std[:,kk]).flatten(), 
                         color='b', alpha=0.3, edgecolor='none')    
                
        axU1.set_ylabel(flag)
        axU1.tick_params(axis='x', labelrotation=30)
        if kk==0: axU1.set_title(description)

    if track_GPS:
        GPS_Z_valid = track_GPS['GPS_Z_valid']
        err=np.abs(xyz-xyzE)
        if not GPS_Z_valid:
            err[:,2] = 0
        err_median = np.nanmedian(err, axis=0)
        err_mean = np.nanmean(err, axis=0)
        err_rms = np.sqrt(np.nanmean(err**2, axis=0))
                
        axB0 = fig.add_subplot(gs[1, 0])
        flierprops = dict(marker='+', markerfacecolor='red', markeredgecolor='red')
        err_noNan = [err[:,j][~np.isnan(err[:,j])] for j in range(err.shape[1])]
        axB0.boxplot(err_noNan,flierprops=flierprops, vert=True)
        axB0.set_ylabel('error', rotation=90)
        axB0.set_xticklabels(["x","y","z"])
        
        for kk in range(3):
            flag=['x','y', 'z'][kk]
            axB1  = fig.add_subplot(gs[1, kk+1])
            axB1.plot(t_datetime,err[:,kk],'b-o', markersize=1)
            axB1.set_ylabel(flag+' error', rotation=90)
            axB1.tick_params(axis='x', labelrotation=30)
            axB1.set_title("Median:%.2f, Mean:%.2f, RMS:%.2f"
                           %(err_median[kk], err_mean[kk], err_rms[kk]))

    if figTitle is not None:
        plt.suptitle(figTitle,x=0.5, y=0.92,fontsize=14)  
        
    if fout is not None:
        plt.savefig(fout, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
    if close:
        plt.close()
    if testInfo is not None:
        periods, names=testInfo['periods'], testInfo['names']
        for (test_starttime, test_endtime), name in zip(periods, names):
            trimed_trackNN = Model.trimTrack(track_NN, test_starttime, test_endtime)
            plot_track_OneFigureEqual(trimed_trackNN, track_GPS=track_GPS, figTitle=figTitle,
                         fout=fout[:-4]+f"_{name}.jpg" if fout is not None else None,
                         close=True, method=method)    
    return 

