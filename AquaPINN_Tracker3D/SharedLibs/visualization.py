# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:09:45 2024

@author: chen096
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import datetime
datetime_NegInf = datetime.datetime(1699,1,1)
datetime_PosInf = datetime.datetime(2222,12,31)


def showDetections(dectectionFile, tagType='Tag', outputdir='.', 
                   time_start=None, time_end=None, timeZone=None):
    time_start = time_start or datetime_NegInf
    time_end = time_end or datetime_PosInf
    with open(dectectionFile, 'rb') as f:
        dectections=pickle.load(f)
    for i,dectection in enumerate(dectections):
        msg = dectection['decodes']
        if msg is None:
            continue
        phoneIDs = msg[:,3].astype(int)
        t_num=msg[:,0]-719529
        t_datetime=pd.to_datetime(t_num, unit='D').round('s') 
        if timeZone is not None:
            t_datetime=t_datetime.tz_localize('UTC').tz_convert(timeZone).tz_localize(None)
        ind = (t_datetime >= time_start) & (t_datetime <= time_end)
        plt.figure()
        plt.title("Detection over time")
        plt.plot(t_datetime[ind], phoneIDs[ind], 'o',markersize=2)
        plt.tick_params(axis='x', labelrotation=30)
        plt.ylabel("Hydrophone Index")
        plt.yticks(np.unique(phoneIDs), labels=np.unique(phoneIDs))
        filename = f"{tagType}{i}.jpg"
        file = os.path.join(outputdir, filename)        
        plt.savefig(file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        