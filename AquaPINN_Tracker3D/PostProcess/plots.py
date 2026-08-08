# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.image as mpimg

datetime_NegInf = pd.to_datetime("1699-1-1")
datetime_PosInf = pd.to_datetime("2222-12-31") 

scaleFactor =1.4


colors=['g', 'y','b', 'c']

_cache = {}
def load_solution_single(solutions_csv, xyz_base, start_time, end_time, PRI: float = 2.0, stdThreshold: float = 3.0):
    start_time = start_time if start_time is not None else datetime_NegInf
    end_time   = end_time if end_time is not None else datetime_PosInf
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)     
    if solutions_csv is None: return None
    start_time = start_time if start_time is not None else datetime_NegInf
    end_time   = end_time if end_time is not None else datetime_PosInf
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    df= pd.read_csv(solutions_csv)
    df.columns = df.columns.str.lower()
    if 'z' not in df.columns:
        df['z'] = np.nan 
    
    df['t_datetime'] = pd.to_datetime(df['datetime'])
    df['t_num'] = np.array([t.timestamp() for t in df['t_datetime']])
    if 'AML' in solutions_csv: # AML specially for YorkHavev AML results
        pass #df['t_datetime'] += pd.Timedelta(hours=-4)
    if 'YAPS' in solutions_csv:
        pass #df['t_datetime'] += pd.Timedelta(hours=-4)
    df=df[(df['t_datetime']<=end_time) & (df['t_datetime']>=start_time)]
    if 'xerror' in df.columns:
        df = df.rename(columns={"xerror": "x error", "yerror": "y error", "zerror": "z error"})  # type: ignore[call-overload]
    if 'x error' not in df.columns:
        df['x error'] = np.zeros_like(df['x'])
        df['y error'] = np.zeros_like(df['y'])
        df['z error'] = np.zeros_like(df['z'])
    
    if df.shape[0]<5:
        return None

    
    # ind = df['t_datetime']<datetime_NegInf
    # for ts,te in validDate:
    #     ind=ind | ( (df['t_datetime']<=pd.to_datetime(te)) & (df['t_datetime']>=pd.to_datetime(ts)) )
    # df = df[ind]
        
    track={}
    track['XYZ'] =df[['x','y','z']].values-xyz_base
    
    track['XYZ_error'] = df[['x error','y error','z error']].values
    track['t_datetime'] = df['t_datetime']
    track['t_num'] = df['t_num'].values
    track['mask']       = np.ones(len(track['t_datetime']))>0
    track['XYZ_std']    = np.zeros_like(track['XYZ'])
    if "x std" in df.columns:
        std = np.maximum(df['x std'].values, df['y std'].values)
        track['mask'] = std < stdThreshold
        track['XYZ_std'] = df[['x std','y std','z std']].values
    N  = np.sum(track['mask'])
    dt = track['t_num'].max()-track['t_num'].min()
    track['npoint'] = N
    #track['npoint_ideal'] = (dt+1e-10)/PRI
    dt = np.diff(track['t_num'])
    nP_interval = np.round(dt/PRI)
    nP_interval = np.maximum(nP_interval, 1)
    nP_interval[nP_interval>10] = 1
    track['npoint_ideal'] = np.sum( nP_interval) + 1
    track['PRI']=PRI
    return track


def load_solution(solutions_csv, xyz_base, start_time, end_time, PRI: float = 2.0, stdThreshold: float = 1.5e10):
    if solutions_csv is None: return None
    if not isinstance(start_time, (list,tuple)):
        return load_solution_single(solutions_csv, xyz_base, start_time, end_time, PRI=PRI, stdThreshold=stdThreshold)
    track_all = {'XYZ': np.zeros((0,3)),
                 'XYZ_error':np.zeros((0,3)),
                 'XYZ_std':np.zeros((0,3)),
                 'mask':np.array([], dtype=bool),
                 't_datetime':np.array([], dtype='datetime64[ns]'),
                 't_num':np.zeros((0)),
                 'npoint':0,
                 'npoint_ideal':0,  
                 'PRI':2,
                 }
    dummy     = {'XYZ': np.zeros((1,3))*np.nan,
                 'XYZ_error':np.zeros((1,3))*np.nan,
                 'XYZ_std':np.zeros((1,3))*np.nan,
                 'mask':np.array([False], dtype=bool),
                 't_datetime':np.array(['NaT'], dtype='datetime64[ns]'),
                 't_num':np.zeros((1))*np.nan,                
                 }
    for st,et in zip(start_time, end_time):           
        track=load_solution_single(solutions_csv, xyz_base, st, et, PRI=PRI, stdThreshold=stdThreshold)
        if track is None: continue
        for k in dummy.keys():
           track_all[k]  = np.concatenate((track_all[k], dummy[k],track[k]))
        for k in ['npoint', 'npoint_ideal']:
            track_all[k] += track[k]
        track_all['PRI'] = track['PRI']
    return track_all
        
def trim_GPS(track_GPS, start_time, end_time):
    if track_GPS is None:
        return None
    if not isinstance(start_time,(list,tuple)):
        start_time=[start_time]
        end_time  = [end_time]
    track_all = {'XYZ': np.zeros((0,3)),
                 't_datetime':np.array([], dtype='datetime64[ns]'),
                 't_num':np.zeros((0)),
                 'GPS_Z_valid':False,                  
                 }
    dummy     = {'XYZ': np.zeros((1,3))*np.nan,
                 't_datetime':np.array(['NaT'], dtype='datetime64[ns]'),
                 't_num':np.zeros((1))*np.nan,                
                 }        
    for st,et in zip(start_time, end_time):
        st = st if st is not None else datetime_NegInf
        et   = et if et is not None else datetime_PosInf
        st = pd.to_datetime(st)
        et = pd.to_datetime(et)          
        tG_datetime=track_GPS['t_datetime']
        ind = (tG_datetime>=st) &(tG_datetime<=et)
        for k in dummy.keys():
           track_all[k]  = np.concatenate((track_all[k], dummy[k],track_GPS[k][ind]))
        for k in ['GPS_Z_valid',]:
            track_all[k] = track_GPS[k]
    return track_all

def getErrorFromRefTrack(track,track_ref):
    track['XYZ_error']=np.zeros_like(track['XYZ'])*np.nan
    ai,bi=find_approx_match_indices(track['t_num'], track_ref['t_num'], tol=0.8)
    if len(ai)>0:
        track['XYZ_error'][ai] = track['XYZ'][ai]-track_ref['XYZ'][bi]
    PRI=track_ref['PRI']
    track['npoint_ideal'] = track['npoint_ideal']*track['PRI']/PRI
    track['PRI']=PRI
    return track

def add_background(ax, image_path,
                   mapping1, mapping2,
                   origin='upper', alpha: float = 1.0):
    pixel_point1, xy_point1 =mapping1
    pixel_point2, xy_point2 =mapping2

    img = mpimg.imread(image_path)
    height, width = img.shape[0], img.shape[1]
    px1, py1 = pixel_point1
    x1, y1 = xy_point1
    px2, py2 = pixel_point2
    x2, y2 = xy_point2
    scale_x = (x2 - x1) / (px2 - px1)
    scale_y = (y2 - y1) / (py2 - py1)
    offset_x = x1 - scale_x * px1
    offset_y = y1 - scale_y * py1
    x_min = offset_x
    x_max = scale_x * width + offset_x
    y_min = offset_y
    y_max = scale_y * height + offset_y
    extent = [x_min, x_max, y_max, y_min]
    ax.imshow(img, extent=extent, origin=origin)
    return extent
    #ax.invert_yaxis()
    

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

    

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
import numpy as np

# 创建五角星路径
def create_pentagram_path(radius=10.0):
    angles = np.linspace(0, 2 * np.pi, 11)[:-1]
    outer_radius = radius
    inner_radius = radius * 0.382  # 黄金比例近似值
    verts = []
    for i in range(10):
        r = outer_radius if i % 2 == 0 else inner_radius
        angle = angles[i]
        verts.append((r * np.cos(angle), r * np.sin(angle)))
    verts.append(verts[0])
    codes = [Path.MOVETO] + [Path.LINETO]*9 + [Path.CLOSEPOLY]
    return Path(verts, codes)



def plot_track_with_Background(solutions_csv, phoneInfo, start_time=None, end_time=None,
                              track_GPS=None, figTitle=None,
                              fout=None, close=True,
                              img_path=None, pixel2Phone1=None, pixel2Phone2=None,
                              alpha: float = 1.0,
                              linestyle='-'):
    assert img_path is not None and pixel2Phone1 is not None and pixel2Phone2 is not None   
    phoneIDs = list(phoneInfo['IDs'])
    xyz_base = phoneInfo['ENE'][0:1]  
    phoneLoc = phoneInfo['ENE']-xyz_base
    phoneID1 = pixel2Phone1[1]
    phoneID2 = pixel2Phone2[1]    
    mapping1=(pixel2Phone1[0], phoneLoc[phoneIDs.index(phoneID1),:2])
    mapping2=(pixel2Phone2[0], phoneLoc[phoneIDs.index(phoneID2),:2])
    
    track_GPS = trim_GPS(track_GPS, start_time, end_time)
    
    solutions={}
    for name, solution_csv in solutions_csv.items():
        track=load_solution(solution_csv, xyz_base, start_time, end_time)  
        solutions[name]=track
    
    fig = plt.figure(figsize=(10, 8))
    nc=1
    gs = gridspec.GridSpec(4,4*nc, figure=fig, wspace=0.2, hspace=0.4,)
    axU0 = fig.add_subplot(gs[:,:])  #
    extent=add_background(axU0, img_path,mapping1=mapping1,mapping2=mapping2,
                       origin='upper', alpha=alpha)
    if track_GPS is not None:
        xyzG = track_GPS['XYZ']-xyz_base
        axU0.plot(xyzG[:,0], xyzG[:,1], 'r-o', markersize=4, label='GPS')
    axU0.plot(phoneLoc[:,0],phoneLoc[:,1],'rp',
              markersize=10*scaleFactor, label='phones')
    for i in range(len(phoneLoc)):
        axU0.text(phoneLoc[i,0],phoneLoc[i,1]-1, f"{phoneIDs[i]}",
                  ha='center', va='bottom', fontsize=8*scaleFactor)    
    for kk, (name, track) in enumerate(solutions.items()):
        if track is None: continue
        xyz=track['XYZ']
        labelNN=f"{name}"        
        axU0.plot(xyz[:,0],xyz[:,1],colors[kk]+'o',
                  linestyle=linestyle,
                  label=labelNN,markersize=3)

    # phone_xrange=[phoneLoc[:,0].min(), phoneLoc[:,0].max()]
    # phone_yrange=[phoneLoc[:,1].min(), phoneLoc[:,1].max()]
    
      
    axU0.set_xlabel('x', fontsize=14*scaleFactor)
    axU0.set_ylabel('y', fontsize=14*scaleFactor)
    axU0.set_xlim(extent[0], extent[1])
    axU0.set_ylim(extent[2], extent[3])
    axU0.set_aspect('equal')
    axU0.tick_params(axis='x', labelsize=12*scaleFactor)
    axU0.tick_params(axis='y', labelsize=12*scaleFactor)
    axU0.legend(loc='best', fontsize=14*scaleFactor)    


    if figTitle is not None:
        plt.suptitle(figTitle,x=0.5, y=0.92,fontsize=14*scaleFactor)  
        
    if fout is not None:
        plt.savefig(fout, dpi=600, bbox_inches='tight', facecolor='white')
    if close:
        plt.close()
    return 


def plot_track_OneFigureEqual(solutions_csv, phoneInfo, start_time=None, end_time=None,
                              track_GPS=None, figTitle=None,
                              fout=None, close=True, PRI: float = 2.0):
    dim = 2  
    phoneIDs = phoneInfo['IDs']
    xyz_base = phoneInfo['ENE'][0:1]  
    phoneLoc = phoneInfo['ENE']-xyz_base
    solutions={}
    for name, solution_csv in solutions_csv.items():
        track=load_solution(solution_csv, xyz_base, start_time, end_time, PRI=PRI)  
        solutions[name]=track
    track_GPS = trim_GPS(track_GPS, start_time, end_time)
    if track_GPS is not None:
        xyzG = track_GPS['XYZ']-xyz_base
        tG_datetime = track_GPS['t_datetime']
    
    
    fig = plt.figure(figsize=(21, 8))
    nc=1
    gs = gridspec.GridSpec(2,4*nc, figure=fig, wspace=0.2, hspace=0.4)
    
    
    axU0 = fig.add_subplot(gs[0,0])  # 
    if track_GPS is not None:
        axU0.plot(xyzG[:,0],xyzG[:,1],'r-', label='GPS')
    axU0.plot(phoneLoc[:,0],phoneLoc[:,1],'rp',markersize=10*scaleFactor, label='phones')
    for i in range(len(phoneLoc)):
        axU0.text(phoneLoc[i,0],phoneLoc[i,1]-1, f"{phoneIDs[i]}",
                  ha='center', va='bottom', fontsize=8*scaleFactor)        
    for kk, (name, track) in enumerate(solutions.items()):
        if track is None: continue
        xyz=track['XYZ']
        mask=track['mask']
        xyz_std = track['XYZ_std'] 
        efficiency= track['npoint']/track['npoint_ideal']
        labelNN=f"{name},{efficiency*100:0.1f}%"        
        axU0.plot(xyz[mask,0],xyz[mask,1],colors[kk]+'o',label=labelNN,markersize=3*scaleFactor)  
        addTrajectoryErrorband(axU0,xyz[:,0],xyz[:,1],xyz_std[:,0], xyz_std[:,1], color=colors[kk], alpha=0.3)
      
    axU0.set_xlabel('x')
    axU0.set_ylabel('y')
    axU0.legend(loc='best', fontsize=12) #*scaleFactor)
    #axU0.set_title("Tracking number:%d, efficiency:%.2f%%"%(xyz.shape[0], track_NN['efficiency']*100))
    
    for j in range(dim):
        axU1 = fig.add_subplot(gs[0,j+1])  # 
        if track_GPS is not None:
            axU1.plot(tG_datetime,xyzG[:,j],'r-', label='GPS')
        for kk, (name, track) in enumerate(solutions.items()):
            if track is None: continue
            t_datetime=track['t_datetime']
            xyz=track['XYZ']        
            xyz_std = track['XYZ_std']        
            mask=track['mask']
            labelNN=f"{name}"
            axU1.plot(t_datetime[mask],xyz[mask,j],colors[kk]+'o',label=labelNN,markersize=3*scaleFactor)
            axU1.fill_between(t_datetime,
                             (xyz[:,j] - 2 * xyz_std[:,j]).flatten(),
                             (xyz[:,j] + 2 * xyz_std[:,j]).flatten(),
                             color=colors[kk], alpha=0.2)
            
        s_dim = ['x', 'y', 'z'][j]
        axU1.set_ylabel(s_dim)
        axU1.tick_params(axis='x', labelrotation=30)
        
    if track_GPS:
        axB0 = fig.add_subplot(gs[1, 0])
        flierprops = dict(marker='+', markerfacecolor='red', markeredgecolor='red')
        err_noNan=[]
        labels=[]
        for j in range(dim):
            for kk, (name, track) in enumerate(solutions.items()):
                if track is None: continue
                t_datetime=track['t_datetime']
                mask=track['mask']
                error=np.abs(track['XYZ_error'])
                err_noNan.append(error[mask,j][~np.isnan(error[mask,j])])
                # labels.append(['x', 'y', 'z'][j]+(name[0]+('+'if name[-1]=='+' else '')))
                labels.append(['x', 'y', 'z'][j]+ name)
        if err_noNan:
            axB0.boxplot(err_noNan, flierprops=flierprops, vert=True)
            axB0.set_ylabel('error', rotation=90)
            axB0.set_xticklabels(labels)
            axB0.tick_params(axis='x', labelrotation=30)
            all_err = np.concatenate(err_noNan)
            if len(all_err) > 0 and np.median(all_err) < 5:
                axB0.set_ylim(0, 10)
            current_ylim = axB0.get_ylim()
            if current_ylim[1] > 100:
                axB0.set_ylim(0, 100)
               
        for j in range(dim):
            axB1  = fig.add_subplot(gs[1, j+1])
            title = 'RMS '
            median = None
            for kk, (name, track) in enumerate(solutions.items()):
                if track is None: continue
                t_datetime = track['t_datetime']
                mask  = track['mask']
                error = np.abs(track['XYZ_error'])
                axB1.plot(t_datetime[mask], error[mask, j], colors[kk]+'o', markersize=3*scaleFactor)
                median = np.nanmedian(error[mask, j])
                RMS    = np.sqrt(np.nanmean(error[mask, j]**2))
                title += f"{name}:{RMS:.2f} "
            s_dim = ['x', 'y', 'z'][j]
            axB1.set_ylabel(f'{s_dim} error', rotation=90)
            axB1.tick_params(axis='x', labelrotation=30)
            if median is not None and median < 5:
                axB1.set_ylim(0, 10)
            axB1.set_title(title)
            current_ylim = axB1.get_ylim()
            axB1.set_ylim(0, current_ylim[1])

    if figTitle is not None:
        plt.suptitle(figTitle,x=0.5, y=0.92,fontsize=14*scaleFactor)  
        
    if fout is not None:
        plt.savefig(fout, dpi=600, bbox_inches='tight', facecolor='white')
    if close:
        plt.close()
    return 
def find_approx_match_indices(a, b, tol=1e-5):
   a = np.asarray(a)
   b = np.asarray(b)
   a_sort_idx = np.argsort(a)
   b_sort_idx = np.argsort(b)
   a_sorted = a[a_sort_idx]
   b_sorted = b[b_sort_idx]
   i = j = 0
   a_result = []
   b_result = []
   while i < len(a_sorted) and j < len(b_sorted):
       diff = a_sorted[i] - b_sorted[j]
       if abs(diff) < tol:
           a_result.append(a_sort_idx[i])
           b_result.append(b_sort_idx[j])
           i += 1
           j += 1
       elif diff < -tol:
           i += 1
       else:
           j += 1
   return np.array(a_result), np.array(b_result)

def plot_track_XYZ(solutions_csv, phoneInfo, start_time=None, end_time=None,
                              track_GPS=None, figTitle=None,
                              fout=None, close=True, PRI: float = 2.0, ref_method=None):
    dim = 3  
    phoneIDs = phoneInfo['IDs']
    xyz_base = phoneInfo['ENE'][0:1]  
    phoneLoc = phoneInfo['ENE']-xyz_base
    solutions={}
    for name, solution_csv in solutions_csv.items():
        track=load_solution(solution_csv, xyz_base, start_time, end_time, PRI=PRI)  
        solutions[name]=track        
    track_GPS = trim_GPS(track_GPS, start_time, end_time)        
    if track_GPS is not None:
        xyzG = track_GPS['XYZ']-xyz_base
        tG_datetime = track_GPS['t_datetime']    
        
    fig = plt.figure(figsize=(21, 8))
    nc=1
    gs = gridspec.GridSpec(3,4*nc, figure=fig, wspace=0.2, hspace=0.02)

    track_ref=None
    if ref_method is not None and ref_method in solutions:
        track_ref=solutions[ref_method]
    for name, track in solutions.items():
        if track is None: continue
        if track_ref is not None:
            solutions[name] = getErrorFromRefTrack(track, track_ref)     
    for j in range(dim):
        axU1 = fig.add_subplot(gs[j,:])  #     
        if track_GPS is not None:
            axU1.plot(tG_datetime,xyzG[:,j],'r-', label='GPS')         
        for kk, (name, track) in enumerate(solutions.items()):
            if track is None: continue
            t_datetime=track['t_datetime']
            xyz=track['XYZ']        
            xyz_std = track['XYZ_std']        
            mask=track['mask']
            labelNN=f"{name} " # ", {len(t_datetime[mask])}"                 
            if 'XYZ_error' in track and name!=ref_method and ref_method in solutions: #np.any(np.isfinite(track['XYZ_error'][:,:2])):
                err= np.abs(track['XYZ_error'])
                labelNN += f': Median={np.nanmedian(err[:,j]):.2f}, RMS={np.nanstd(err[:,j]):.2f}'
            axU1.plot(t_datetime[mask],xyz[mask,j],colors[kk]+'o',label=labelNN,markersize=1*scaleFactor)
            # axU1.fill_between(t_datetime,
            #                  (xyz[:,j] - 2 * xyz_std[:,j]).flatten(),
            #                  (xyz[:,j] + 2 * xyz_std[:,j]).flatten(),
            #                  color=colors[kk], alpha=0.2)   
        s_dim = ['x', 'y', 'z'][j]
        axU1.legend(loc='best', ncol=3, fontsize=14*scaleFactor, markerscale=4)
        axU1.set_ylabel(s_dim, fontsize=14*scaleFactor)
        axU1.tick_params(axis='x', labelrotation=30, labelsize=12*scaleFactor)
        axU1.tick_params(axis='y', labelsize=12*scaleFactor)
        if j<dim-1:
            axU1.get_xaxis().set_visible(False)
            
    if figTitle is not None:
        plt.suptitle(figTitle,x=0.5, y=0.92,fontsize=14*scaleFactor)  
        
    if fout is not None:
        plt.savefig(fout, dpi=600, bbox_inches='tight', facecolor='white')
    if close:
        plt.close()
    return 


def getMetrics(solutions_csv, phoneInfo, start_time=None, end_time=None,
               PRI: float = 2.0):
    xyz_base = phoneInfo['ENE'][0:1]  
    solutions={}
    for name, solution_csv in solutions_csv.items():
        track=load_solution(solution_csv, xyz_base, start_time, end_time, PRI=PRI)  
        solutions[name]=track
    metrics={}
    for kk, (name, track) in enumerate(solutions.items()):  
        if track is not None: 
            metrics[name] = {'XYZ_error': track['XYZ_error'],
                             'npoint':track['npoint'],
                             'npoint_ideal':track['npoint_ideal'],
                             }
        else:
            metrics[name] = {'XYZ_error': np.zeros((0,3)),
                             'npoint':0,
                             'npoint_ideal':0,
                             }  
            
    return metrics

