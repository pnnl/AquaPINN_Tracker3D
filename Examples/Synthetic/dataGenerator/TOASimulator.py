# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:22:06 2025

@author: chen096
"""
import numpy as np
from scipy.interpolate import interp1d
from math import sin,cos
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle

T=8 # temperature
soundSpeed = (  1.402385E3 + 5.038813* T -5.799136E-2 * T**2 
                +3.287156E-4 * T**3  - 1.398845E-6* T**4 + 2.78786E-9* T**5 )

dis    = np.array([0,11.3,26.4,42.8,62.6,82.3,99.3,120.8]);
prob   = np.array([1.0,0.895,0.9075,0.7595,0.5245,0.383,0.1045,0.037]);
pro_dis = interp1d(dis, prob,kind='linear',
                   bounds_error=False, fill_value=0)


def getDatetime(t):
    refDate=pd.to_datetime("2023-10-16 15:30:00")
    time=refDate.timestamp()/3600/24  + t/3600/24 + 719529 # convert matlab time
    return time

def ENE2XYZ(ENE, ENE0=None, theta=None):
    xyz = ENE -ENE0
    m = np.array([[cos(theta), -sin(theta), 0],
                  [sin(theta),  cos(theta), 0],
                  [         0,           0, 1]]).T
    xyz = xyz@m
    return xyz

def extractTrack(track, PRI):
    t=track['t']
    xyz = track['XYZ']
    vel = track['vel']    
    tNew= np.arange(t[0], t[-1], PRI)
    f = interp1d(t, xyz,axis=0,kind='linear',
                   bounds_error=False, fill_value=0)
    xyz_new = f(tNew)
    f = interp1d(t, vel,axis=0,kind='linear',
                   bounds_error=False, fill_value=0)    
    vel_new = f(tNew)
    trackNew={'t':tNew, 'XYZ':xyz_new, 'vel':vel_new}
    return trackNew

def getTrackCRLB(track, phoneLoc, noise=0):
    from getCRLB import compute_crlb_toa_3d
    xyz = track['XYZ']
    t = track['t']
    hydro_pos = phoneLoc
    sigma_t = noise  # Convert noise to distance
    c = soundSpeed
    
    crlb=np.zeros_like(xyz)
    for i, xyzi in enumerate(xyz):
        crlb[i] = compute_crlb_toa_3d(xyzi, hydro_pos, sigma_t, c)
    track['CRLB'] = np.sqrt(crlb)
    return track

def transformTrack(track, xyz0, theta):
    xyz = track['XYZ']
    xyz = ENE2XYZ(xyz, xyz0, theta)
    track['XYZ'] = xyz + xyz0
    return track
    
def generateTagDecodes(track, phoneLoc,
                    noise=0,
                    fout=None, applyMissing=True, noise_outlier=0, outlierRatio=2):  
    nPhone=len(phoneLoc)
    xyz=track['XYZ']
    top=np.tile(track["t"][:,None], (1,nPhone))
    phoneIDs=np.tile(np.arange(nPhone)[None,:]+1, (len(top),1))
    

    r=np.linalg.norm(xyz[:,None,:]-phoneLoc[None,:,:], axis=-1)
    toa=top+r/soundSpeed+np.random.randn(*r.shape)*noise
    mask = np.random.rand(*r.shape)<(pro_dis(r) if applyMissing else 1.0)
    
    date=getDatetime(top)
    time=getDatetime(toa)
    SNR=np.zeros_like(r)
    decodes=np.stack([ d[mask] for d in [date,time,SNR,phoneIDs] ], axis=1)
    noise_random_outlier = np.random.randn(*decodes[:, :2].shape)*noise_outlier
    decodes[:, :2] += (np.abs(noise_random_outlier)>outlierRatio*noise_outlier)*noise_random_outlier/3600/24
    tagDecodes=[{'decodes':decodes}]
 
    if fout is not None:
        with open(fout, 'wb') as f:
            pickle.dump(tagDecodes, f)    
    return tagDecodes

def determineTheta(ENE):
    U, S, Vt = np.linalg.svd(ENE[:,:2]-np.mean(ENE[:,:2], axis=0)[None,:])
    yAxisOld = np.array([0,1])
    yAxisNew = Vt[0]
    cos_theta_y = np.dot(yAxisOld, yAxisNew)/np.linalg.norm(yAxisNew)
    if cos_theta_y<0:
        cos_theta_y *=-1 
        yAxisNew *= -1
    theta_y = np.arccos(np.clip(cos_theta_y, -1.0, 1.0))
    cross_product_y = np.cross(yAxisOld, yAxisNew)
    if cross_product_y < 0:
        theta_y *= -1
    print(f"Rotate angle {theta_y} to the main eigendirection")
    return -theta_y


def loadPhoneLocFile(phoneLocfile, badPhones=[], return_df=False):
    "load hydrophone location from a csv file"
    phone = pd.read_csv(phoneLocfile)
    cnames = [re.sub('[^A-Za-z0-9]+', '', c) for c in phone.columns]
    cnames = [c[:1].lower() + c[1:] for c in cnames]
    phone.columns = cnames
    for i, name in enumerate(phone.columns):
        name_lower=name.lower()
        if 'northing' in name_lower:
            cnames[i]='northing'
        elif 'easting' in name_lower:
            cnames[i]='easting'
        elif 'elevation' in name_lower or 'depthft' in name_lower:
            cnames[i]='elevation'
        elif 'nodesn' in name_lower:
            cnames[i]='nodesn'
        elif 'beacon' in name_lower:
            cnames[i]='beacon'
    phone.columns = cnames
    # feet to meter
    phone_old = phone.copy()
    phone[['northing','easting','elevation']] = phone[['northing','easting','elevation']]*0.3048
    ENE  = phone[['easting','northing','elevation']].values
    pIDs = list(phone['phoneID'].values)
    ENE0 = ENE.mean(axis=0)[None,:]
    ind_goodPhones=[i for i, pID in enumerate(pIDs) if pID not in badPhones]
    SN2ID = {int(sn):int(i) for sn, i in  phone[['nodesn','phoneID']].values}
    theta = determineTheta(ENE)
    phoneInfo = {'XYZ':ENE2XYZ(ENE, ENE0, theta)[ind_goodPhones],
                 'IDs_all':phone['phoneID'].values.astype(int),
                 'IDs':phone['phoneID'].values.astype(int)[ind_goodPhones],
                 'ENE0':ENE0,
                 'theta':theta,
                 'SN2ID':SN2ID,
                 "AttachedBeacons": ([None]*len(SN2ID) if 'beacon' not in phone.columns 
                                     else 
                                     [ None if p is np.nan else p for p in phone['beacon'].values] ),
                 'badPhones':badPhones,
                 'dataFrame':phone_old}
    return phoneInfo   



def XYZ2ENE(XYZ, ENE0=0, theta=0):    
    theta *= -1
    m = np.array([[cos(theta), -sin(theta), 0],
                  [sin(theta),  cos(theta), 0],
                  [         0,           0, 1]]).T
    xyz = XYZ@m
    ENE = xyz + ENE0
    return ENE 

def generateGPSFile(track,fout, ENE0=0, theta=0): 
    ene=track['XYZ']
    ENE=XYZ2ENE(ene, ENE0, theta) 
    # ENE = ENE/0.3048
    Datetime=pd.to_datetime(getDatetime(track["t"])-719529, unit='D').round('s')
    df=pd.DataFrame(ENE,columns=["Easting","Northing","Elevation"])
    df['Datetime']=Datetime
    if 'CRLB' in track:
        df['CRLB_X']=track['CRLB'][:,0]
        df['CRLB_Y']=track['CRLB'][:,1]
        df['CRLB_Z']=track['CRLB'][:,2] 
    df.to_csv(fout, index=False, date_format='%m/%d/%Y %H:%M:%S')
    return
def loadGPS(GPS_file, useLocalXYZ=False, phoneInfo=None):
    trackGPS = pd.read_csv(GPS_file)
    cnames=trackGPS.columns.values
    GPS_Z_valid = False
    use_feetUnit=False
    for i, name in enumerate(trackGPS.columns):
        name_lower=name.lower()
        if 'northing' in name_lower:
            cnames[i]='northing'
            if "ft" in name_lower or "feet" in name_lower:
                use_feetUnit = True
        elif 'easting' in name_lower:
            cnames[i]='easting'
        elif 'elevation' in name_lower:
            cnames[i]='elevation'
            GPS_Z_valid = True   
        elif 'date' in name_lower and 'time' in name_lower:
            cnames[i]='Datetime'
    trackGPS.columns=cnames
    if not GPS_Z_valid:
        trackGPS['elevation']=1.5
    if 'Date' in trackGPS.columns and 'Time' in trackGPS.columns:
        t =pd.to_datetime(trackGPS['Date'] +' ' +trackGPS['Time'], format='%m/%d/%Y %H:%M:%S')
    else:
        t =pd.to_datetime(trackGPS['Datetime'], format='%m/%d/%Y %H:%M:%S')
    _,ind=np.unique(t, return_index=True)
    t_num = np.array([ ti.timestamp() for ti in t])
    trackGPS['t_datetime'] = t
    trackGPS['t_num'] = t_num
    NEE  = trackGPS[['easting','northing','elevation']].values
    if use_feetUnit:
        NEE*=0.3048
    trackGPSDict = {'XYZ':NEE[ind] if not useLocalXYZ else ENE2XYZ(NEE, phoneInfo['ENE0'], phoneInfo['theta'])[ind],
                    't_datetime':t[ind],
                    't_num':t_num[ind],
                    'GPS_Z_valid':GPS_Z_valid}
    return trackGPSDict


def plotTrack(track, phoneLoc=None,vel=None, title=None, fout=None):
    fig = plt.figure(figsize=(12,4))
    vel = track['vel']
    t   = track['t']
    xyz = track['XYZ']
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(xyz[:,0], xyz[:,1], xyz[:,2], marker='.', color='blue')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D view')
    if phoneLoc is not None:
        ax1.plot(phoneLoc[:,0],phoneLoc[:,1],phoneLoc[:,2],
                 'rp',markersize=10)
    #plt.axis('equal')
    
    ax2 = fig.add_subplot(132) 
    ax2.plot(xyz[:,0], xyz[:,1], marker='.', color='blue')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('2D view')    
    if phoneLoc is not None:
        ax2.plot(phoneLoc[:,0],phoneLoc[:,1],
                 'rp',markersize=10)
    plt.axis('equal')
    
    ax3 = fig.add_subplot(133) 
    ax3.plot(t, np.linalg.norm(vel, axis=1), marker='.', color='blue')
    ax3.set_xlabel('t')
    ax3.set_ylabel('vel')
    ax3.set_title('Velocity')
    
    if title is not None:
        fig.suptitle(title, fontsize=16)
    if fout is not None:
        plt.savefig(fout, dpi=600, bbox_inches='tight', facecolor='white')
    return
