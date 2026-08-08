# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 01:18:11 2024

@author: chen096
"""
import numpy as np
import pandas as pd
from math import sin,cos
import os,re
import pickle
import logging
import utm
from datetime import datetime, timedelta

datetime_NegInf = pd.to_datetime("1699-1-1")
datetime_PosInf = pd.to_datetime("2222-12-31")    


logger = logging.getLogger(__name__)

from .SharedLibs.functions import load_params, replaceFolder

class localizationSystem():
    def __init__(self,):
        self.phoneInfo=None
        self.tagInfo=None
        self.testInfo=None
        return
    def deploy(self, param_file, doSync=True):
        train_params = load_params(param_file)[0] 
        phoneFile=train_params['phoneFile']
        outputFolder=train_params['output_folder']
        self.phoneInfo=self.deployPhone( phoneFile=phoneFile,
                                         badPhones=train_params['badPhones'],
                                         )
        phoneFile_synced = os.path.splitext(phoneFile)[0]+'_synced.csv'
        phoneFile_synced=replaceFolder(phoneFile_synced, outputFolder)             
        if not doSync:
            logger.warning(f'Synced phone location file is founded and loaded: {phoneFile_synced}')
            assert os.path.isfile(phoneFile_synced)
            logger.warning(f'Synced phone location file is founded and loaded: {phoneFile_synced}')
            phoneInfo_synced=self.deployPhone( phoneFile=phoneFile_synced,
                                             theta= self.phoneInfo['theta'],
                                             ENE0 = self.phoneInfo['ENE0'],
                                             badPhones=train_params['badPhones'],
                                              ) 
            self.phoneInfo = phoneInfo_synced
            #self.phoneInfo['XYZ'] = phoneInfo_synced['XYZ']            
        self.deployTag(tagFile=train_params['tagFile'])
        if os.path.isfile(train_params['testFile']):
            self.deployTest(train_params['testFile'])
        
        tz_read=train_params['timeZone']
        assert tz_read in range(-12,14), "timeZone must be integer in [-12,13]"
        tz_read = int(tz_read) # timezone in hours
        self.timeZone_local = tz_read
        return self
    def deployPhone(self, phoneFile, origin_idx=1, theta=None, ENE0=None, badPhones=[], return_df=False):
        "load hydrophone location from a csv file"
        #phone = pd.read_csv('/home/linx882/Lamprey/test_LGR23/data/LGR_hydrophone_configuration_061413_allDamPhones_update_for_2023.csv')
        phone = pd.read_csv(phoneFile)
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
            elif 'nodesn' in name_lower  or 'sn' == name_lower:
                cnames[i]='nodesn'
            elif 'beacon' in name_lower:
                cnames[i]='beacon'
        phone.columns = cnames
        # replace non-numeric sn and phoneID with numeric ones starting from 1000 
        for col in ['nodesn', 'phoneID']:
            if col in phone.columns:
                isNan = phone[col].isna()
                phone.loc[isNan, col] = range(1000, 1000+isNan.sum())
                phone[col] = phone[col].astype(int)     
                    
        
        # feet to meter
        phone_old = phone.copy()
        try:
            phone[['northing','easting','elevation']] = phone[['northing','easting','elevation']]*0.3048
        except:
            phone[['easting', 'northing', 'zone_number', 'zone_letter']] = phone.apply(
            lambda row: pd.Series(utm.from_latlon(row['latitude'], row['longitude'])),
            axis=1)
            phone[['elevation']] = phone[['elevation']]*0.3048
            pass
        ENE  = phone[['easting','northing','elevation']].values
        pIDs = list(phone['phoneID'].values)
        if ENE0 is None:
            ENE0 = ENE.mean(axis=0)[None,:]
        ind_goodPhones=[i for i, pID in enumerate(pIDs) if pID not in badPhones]
        SN2ID = {int(sn):int(i) for sn, i in  phone[['nodesn','phoneID']].values}
        if theta is None:
            theta = self.determineTheta(ENE)
        logger.info(f"ENE0={ENE0}")
        logger.info(f"theta={theta}")        
        xyz   = self.ENE2XYZ(ENE, ENE0, theta)

        phoneInfo = {'XYZ':xyz[ind_goodPhones],
                     'XYZ_all':xyz,
                     'IDs_all':phone['phoneID'].values.astype(int),
                     'IDs':phone['phoneID'].values.astype(int)[ind_goodPhones],
                     'ENE':ENE,
                     'ENE0':ENE0,
                     'theta':theta,
                     'SN2ID':SN2ID,
                     "AttachedBeacons": ([None]*len(SN2ID) if 'beacon' not in phone.columns 
                                         else 
                                         [ None if p is np.nan else p for p in phone['beacon'].values] ),
                     'badPhones':badPhones,
                     'dataFrame':phone_old}
        return phoneInfo   
    def determineTheta(self,ENE):
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
        logger.info(f"Rotate angle {theta_y} to the main eigendirection")
        return -theta_y
        
    def deployTag(self,tagFile):
        tagdf = pd.read_csv(tagFile)
        tagInfo={}
        tagInfo['code']=list(tagdf['Code'].values)
        tagInfo['PRI']=list(tagdf['PRI'].values)
        self.tagInfo=tagInfo
        return tagInfo
    def deployTest(self,testFile):
        test_data = pd.read_csv(testFile)
        cnames = [re.sub('[^A-Za-z0-9]+', '', c) for c in test_data.columns]
        cnames = [c.lower() for c in cnames]
        test_data.columns = cnames            
        test_data['starttime'] = pd.to_datetime(test_data['starttime'], format='%m/%d/%Y %H:%M:%S')
        test_data['endtime']   = pd.to_datetime(test_data['endtime'], format='%m/%d/%Y %H:%M:%S')
        periods = list(test_data[['starttime','endtime']].values)
        names   = list(test_data['name'].values)        
        testInfo ={'periods':periods, 'names':names}      
        self.testInfo=testInfo
        return testInfo    
    def ENE2XYZ(self,ENE, ENE0=None, theta=None):
        xyz = ENE -ENE0
        m = np.array([[cos(theta), -sin(theta), 0],
                      [sin(theta),  cos(theta), 0],
                      [         0,           0, 1]]).T
        xyz = xyz@m
        return xyz
    
    def XYZ2ENE(self, XYZ, ENE0=None, theta=None):     
        if ENE0 is None or theta is None:
            ENE0 = self.phoneInfo['ENE0']
            theta = self.phoneInfo['theta']
        theta *= -1
        m = np.array([[cos(theta), -sin(theta), 0],
                      [sin(theta),  cos(theta), 0],
                      [         0,           0, 1]]).T
        xyz = XYZ@m
        ENE = xyz + ENE0
        return ENE 
    
    def loadDecodes(self, decodesFile, converTimeZone=False):
        with open(decodesFile, 'rb') as file:
            decodes=pickle.load(file)  
        from .RawDataProcess.receiver_dataprep import mapNodeCode2ID
        decodes_new=[]
        for p in decodes:
            if p['decodes'] is not None:
                p['decodes']=mapNodeCode2ID(self,p['decodes'])
                if converTimeZone:
                    TOA_t_num= p['decodes'][:,0:1]
                    p['decodes'][:,:2] = TOA_t_num + self.timeZone_local/24
            decodes_new.append(p)
        return decodes_new    
    def getDecodesFile(self,tagFile, flag, newFolder=None):
        file=os.path.splitext(tagFile)[0]+'_' + flag +'Decodes.pickle'
        if newFolder is not None:
            file = replaceFolder(file, newFolder)
        return file
    def getSyncedDecodesFile(self,tagFile, flag, newFolder=None):
        return os.path.splitext(self.getDecodesFile(tagFile, flag, newFolder=newFolder))[0]+'_synced.pickle'


    def applySync2Decodes(self, decodesFile, badPhones=[], syncFile=None, newFolder=None,
                          time_start=None,
                          time_end=None):
        decodesFile_sync=os.path.splitext(decodesFile)[0]+'_synced.pickle'
        if newFolder is not None:
            decodesFile_sync=replaceFolder(decodesFile_sync, newFolder)
        with open(syncFile, "rb") as f: 
            timeSync=pickle.load(f)
        coeff_sync = timeSync['timeSync']  
        syncable_phoneIDs = timeSync['syncable_phoneIDs']
        tmin_sync,tmax_sync=timeSync['tmin'], timeSync['tmax']    
        decodes=self.loadDecodes(decodesFile, converTimeZone=True)
        for iTag, tag in enumerate(decodes):
            info_mat=tag['decodes']
            if info_mat is None:
                continue
            days=info_mat[:,0]
            time=info_mat[:,1]*24*3600 
            phoneIDs, ind_phones = np.unique(info_mat[:,3].astype(int),return_inverse=True)
            t= time
            for i,ID in enumerate(phoneIDs):
                if ID in badPhones:
                    logger.warning(f'bad phone {ID} is not synced')
                    continue
                (a,b,c), t_jump, v_jump=coeff_sync[ID]
                v_jump=np.concatenate((v_jump[:1], np.diff(v_jump)), axis=0 )
                ind=ind_phones==i
                tnorm=(t[ind]-(tmax_sync+tmin_sync)/2)/(tmax_sync-tmin_sync)*2
                Npiece=len(a)
                ind_piece = (tnorm+1)/2*Npiece
                ind_piece = np.maximum(np.minimum(ind_piece, Npiece-1e-10), 0).astype(np.int32)
                a,b,c=[ p[ind_piece] for p in [a,b,c]]
                
                
                t[ind]+=a*tnorm**2+ b*tnorm + c
                toa_jump = (tnorm[:,None]>t_jump[None,:])*v_jump[None,:]
                t[ind] += toa_jump.sum(axis=-1)
                # (a,b,c), t_jump, v_jump=coeff_sync[ID]
                # ind=ind_phones==i
                # tnorm=(t[ind]-(tmax_sync+tmin_sync)/2)/(tmax_sync-tmin_sync)*2
                # t[ind]+=a*tnorm**2+ b*tnorm + c
                # toa_jump = (tnorm[:,None]>t_jump[None,:])*v_jump[None,:]
                # t[ind] += toa_jump.sum(axis=-1)                               
            info_mat[:,1] = t/24/3600
            info_mat[:,0] = t/24/3600
            for i,ID in enumerate(phoneIDs):
                if ID not in badPhones and ID not in syncable_phoneIDs:
                    logger.warning(f'phone {ID} is not synable, remove it from decodes file')
                    ind=info_mat[:,3].astype(int)==ID
                    info_mat[ind,0:2] = 0+719529
            TOA_t_num = info_mat[:,1]-719529
            TOA_t_datetime= pd.to_datetime(TOA_t_num, unit='D')
            ind = (time_start<=TOA_t_datetime) & (TOA_t_datetime<=time_end)
            decodes[iTag]['decodes'] = info_mat[ind]
        
        # if os.path.isfile(decodesFile_sync):
        #     decodesOld=self.loadDecodes(decodesFile_sync, converTimeZone=False)  
        #     for iTag, tag in enumerate(decodesOld):
        #         info_mat=tag['decodes']
        #         TOA_t_num = info_mat[:,1]-719529
        #         TOA_t_datetime= pd.to_datetime(TOA_t_num, unit='D')
        #         ind = (time_start<TOA_t_datetime) | (TOA_t_datetime>time_end)
        #         decodes[iTag]['decodes']  = np.concatenate((info_mat[ind], decodes[iTag]['decodes']), axis=0)                
        #     logger.info("old synced file is overwritten")
        with open(decodesFile_sync, 'wb') as file:
                pickle.dump(decodes, file)
        logger.info(f"synced tag detection file has been saved at {decodesFile_sync} ")        
        return 
    
    def applySync2Phone(self, phoneFile,  syncFile, newFolder=None):
        phoneInfo=self.phoneInfo
        phoneDF=phoneInfo['dataFrame']
        with open(syncFile, "rb") as f: 
            timeSync=pickle.load(f)
        self.phoneInfo['XYZ']=phoneLoc_XYZ = timeSync['phoneLoc']
        phoneIDs = timeSync['phoneIDs']
        phoneID_all=list(phoneDF['phoneID'])
        for i,phoneID in enumerate(phoneIDs):
            ENE=self.XYZ2ENE(phoneLoc_XYZ[i:i+1,:], phoneInfo['ENE0'], phoneInfo['theta']) 
            ENE = ENE[0,:]/0.3048
            phoneDF.loc[phoneID_all.index(phoneID),['easting','northing','elevation']]=ENE
        newFile=os.path.splitext(phoneFile)[0]+'_synced.csv'
        if newFolder is not None:
            newFile=replaceFolder(newFile, newFolder)        
        phoneDF.to_csv(newFile)
        logger.info(f"Corrected phone location file has been saved at {newFile} ")
        return
    
    def loadGPS(self, GPS_file, useLocalXYZ=False):
        file_format = os.path.splitext(GPS_file)[-1].lower()
        assert file_format in ['.pos', '.csv'], f"Unsupported GPS file format: {file_format}, only .pos and .csv are supported"
        phoneInfo = self.phoneInfo
        try:
            trackGPS = self.read_rinex_data(GPS_file)
        except:
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
        trackGPSDict = {'XYZ':NEE[ind] if not useLocalXYZ else self.ENE2XYZ(NEE, phoneInfo['ENE0'], phoneInfo['theta'])[ind],
                        't_datetime':t[ind],
                    't_num':t_num[ind],
                    'GPS_Z_valid':GPS_Z_valid}
        return trackGPSDict



    def read_rinex_data(self, file_path):
        data_lines = []
        if not file_path.endswith(".pos"):
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith('%'):
                        column_names = [col.strip().lower() for col in line.strip().strip('%').split(',')]
                    else:
                        data_lines.append(line)
            column_names[0] = 'date'
            data = pd.DataFrame([x.strip().split(",") for x in data_lines], columns=column_names)
        else:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith('%'):  # Metadata lines start with '%'
                        column_names = [col.strip().lower() for col in line.strip().strip('%').split()]
                    else:
                        # Only keep lines that look like the data rows
                        if re.match(r'\d{4}/\d{2}/\d{2}', line):
                            data_lines.append(line.strip())
            column_names = ['date'] + column_names
            data = pd.DataFrame([x.split() for x in data_lines], columns=column_names)

        if "utc" not in column_names:
            if file_path.endswith(".pos"):
                data['datetime_gpst'] = pd.to_datetime(data['date'] + ' ' + data['gpst'], format='%Y/%m/%d %H:%M:%S.%f')
            else:
                data['datetime_gpst'] = pd.to_datetime(data['date'] + ' ' + data['gpst'], format='%m/%d/%Y %H:%M:%S.%f')
            leap_seconds = 18  # Currently, GPST is 18 seconds ahead of UTC
            data['utc'] = data['datetime_gpst'] - timedelta(seconds=leap_seconds)
            pass
        else:
            if file_path.endswith(".pos"):  
                data['utc'] = pd.to_datetime(data['date'] + ' ' + data['utc']) #, format='%Y/%m/%d %H:%M:%S.%f')
            else:
                data['utc'] = pd.to_datetime(data['date'] + ' ' + data['utc']) #, format='%m/%d/%Y %H:%M:%S.%f')

        data['latitude(deg)'] = pd.to_numeric(data['latitude(deg)'], errors='coerce')
        data['longitude(deg)'] = pd.to_numeric(data['longitude(deg)'], errors='coerce')        
        data['height(m)'] = pd.to_numeric(data['height(m)'], errors='coerce')
        data.rename(columns={'height(m)': 'elevation'}, inplace=True)

        data[['easting', 'northing', 'zone_number', 'zone_letter']] = data.apply(
            lambda row: pd.Series(utm.from_latlon(row['latitude(deg)'], row['longitude(deg)'])),
            axis=1)
          
        #data['utc_offset'] = data.apply(lambda row: approximate_utc_offset(row['longitude(deg)']), axis=1)
        #data['t_datetime'] = data.apply(determine_local_time, axis=1)
        data['t_datetime'] = data['utc'] + pd.Timedelta(hours=self.timeZone_local)

        columns_to_keep = ['t_datetime', 'utc', 'easting', 'northing', 'elevation']
        data = data[columns_to_keep]
        return data
    

def is_pdt(date):
    year = date.year
    march = datetime(year, 3, 1)
    second_sunday_march = march + timedelta(days=(6 - march.weekday() + 7))  # First Sunday + 7 days for second
    november = datetime(year, 11, 1)
    first_sunday_november = november + timedelta(days=(6 - november.weekday()))
    return second_sunday_march <= date < first_sunday_november

def approximate_utc_offset(longitude):
    offset_hours = round(longitude / 15)
    return timedelta(hours=offset_hours)

def convert_utc_to_local(datetime_utc, longitude):
    base_offset = approximate_utc_offset(longitude)
    if is_pdt(datetime_utc):
        local_datetime = datetime_utc + base_offset - timedelta(hours=1)  # PDT is UTC-7
    else:
        local_datetime = datetime_utc + base_offset  # PST is UTC-8
    return local_datetime

def determine_local_time(row):
    """Determine the local time based on whether the datetime is in PDT or PST."""
    datetime_utc = row['utc']
    base_offset  = row['utc_offset']

    # Adjust the offset depending on whether it is PDT or PST
    if is_pdt(datetime_utc):
        # PDT is UTC-7
        local_datetime = datetime_utc + base_offset + timedelta(hours=1)
    else:
        # PST is UTC-8
        local_datetime = datetime_utc + base_offset
    
    return local_datetime
