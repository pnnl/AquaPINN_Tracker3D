import numpy as np
import pickle
import os
import re
import pandas as pd
import json
import glob
import warnings
import gc
from ..SharedLibs.decorators import timing
from .read_recevier_file_csv import read_data_and_extract_SN_binary
# Suppress all warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger(__name__)

__cache={}
def cleanCache():
    __cache.clear()
    
# Function to convert datetime to MATLAB datenum 
## @timing
def dtime2dnum(df):
    df['DateTime'] = df['DateTime'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    # Apply the function to the list of datetime strings
    df['DateTime']= [parse_datetime(date_str) for date_str in df['DateTime']]
    # MATLAB datenum offset for 1970-01-01 (Unix epoch in MATLAB's datenum system)
    MATLAB_DATENUM_OFFSET = 719529  # Number of days betwe
    # Convert datetime to numeric timestamp (seconds since epoch) and convert to days
    df['unix_days'] = df['DateTime'].apply(lambda x: x.timestamp() / (24 * 3600) if pd.notnull(x) else np.nan)
    # Convert Unix days to MATLAB datenum by adding the offset
    df['DTime']  = df['unix_days'] + MATLAB_DATENUM_OFFSET
    df['DTime'] -= df['timeZone']/24
    return df
# Function to handle datetime strings with or without microseconds
# @timing
def parse_datetime(date_str):
	try:
		# Try parsing with microseconds
		return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M:%S.%f')
	except ValueError:
		# If microseconds are missing, parse without them
		return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M:%S')



@timing
def filter_by_tdiff(df):
	# Calculate the time difference between consecutive rows in seconds
	df['time_diff'] = df['DateTime'].diff().dt.total_seconds()
	df['timestamp'] = df['DateTime'].astype('int64') // 10**9
	
	rm_idx_helper = filter_by_diff_helper(df['DateTime'])

	# Drop the rows by index and return the filtered DataFrame
	filtered_df = df.drop(index=rm_idx_helper)

	# Drop the 'time_diff' column as it's no longer needed
	#filtered_df = filte
	return filtered_df

from datetime import datetime
@timing
def filter_by_diff_helper(timestamps):
    # Convert the Series to a list for indexing
    timestamps = list(timestamps)
    rm_idx = []
    n = len(timestamps)
    idx = 0
    while idx < n - 1:
        baseline_ts = timestamps[idx]
        # Find all subsequent rows within threshold
        j = idx + 1
        sidx_3s = []
        # Use total_seconds() for datetime differences
        while j < n and (timestamps[j] - baseline_ts).total_seconds() <= 0.3:
            sidx_3s.append(j)
            j += 1
        if len(sidx_3s) > 0:
            rm_idx.extend(sidx_3s)
            idx = max(sidx_3s) + 1
        else:
            idx += 1
    return rm_idx


@timing
# Function to collect all CSV files from the directory and save data by series numbers
def process_all_csv_files(datafolder, time_start_UTC=None, time_end_UTC=None):
	data_by_serial_number = {}
	# Traverse directory and subdirectories for CSV files
	for root, dirs, files in os.walk(datafolder):
		for file in files:
			if file.lower().endswith('.csv'):
				file_path = os.path.join(root, file)
				if (file_path, time_start_UTC, time_end_UTC) not in __cache:
					df = read_data_and_extract_SN_binary(file_path, time_start_UTC=time_start_UTC, time_end_UTC=time_end_UTC)
					__cache[(file_path, time_start_UTC, time_end_UTC)] = df		
				df = __cache[(file_path, time_start_UTC, time_end_UTC)]
				# Group dataframes by serial number (NodeCode)
				if df.empty: continue # Skip empty dataframes
				serial_number = df['NodeCode'].iloc[0]
				if serial_number not in data_by_serial_number:
					data_by_serial_number[serial_number] = pd.DataFrame()  # Initialize empty DataFrame
				#data_by_serial_number[serial_number].append(df)
				# Concatenate dataframes
				data_by_serial_number[serial_number] = pd.concat([data_by_serial_number[serial_number], df], ignore_index=True)
				# Free up memory by triggering garbage collection
				del df
				gc.collect()
	# Create a list of dictionaries where each dictionary contains a serial number and combined DataFrame
	result_list = [{serial_number: combined_df} for serial_number, combined_df in data_by_serial_number.items()]
	return result_list

# function to get receiver data for tag 
@timing
def get_receiver_data4tag(data_by_serial_number,itag,i):
	ndata=[]
	#data_files = [f for f in glob.glob(datafolder+'/*') if ('.csv' in f or '.CSV' in f)]
	for item in data_by_serial_number:
		for serial, tdf in item.items():
			print(serial)
			logger.info(f"Serial Number: {serial}, DataFrame shape: {tdf.shape}")
			idf = tdf[tdf['TagCode']==itag].reset_index(drop=True).drop_duplicates()
			#  sort data by datetime
			idf = idf.sort_values(by=['DateTime'])
			if len(idf)>0:
				idf['TagID'] = i+1
				#get columns tag ID, detection time, node_code, SigStr
				idf = dtime2dnum(idf)
				# filter out Multipath (with detection times within 0.3 s)
				idf = filter_by_tdiff(idf)
				logger.info('For Tag '+str(itag)+', length after filtering: '+ str(len(idf)))
				idf= idf[['DTime','DTime','SigStr','NodeCode']]
				ndata +=[idf.copy()]
			else:
				logger.info('No receiver data for '+str(itag)+ ' in SN'+str(serial))
	#
	return ndata

# function to compile receiver data for all tags into a dictionary
def mapNodeCode2ID(loc_sys, detections):
	phoneInfo=loc_sys.phoneInfo
	SN2ID=phoneInfo['SN2ID']
	pIDs=detections[:,3].astype(int)
	pID_is_SN = set(pIDs).intersection( set(SN2ID.keys()) )
	ind_clear=np.zeros(0).astype(int)
	if pID_is_SN:
		for k in set(pIDs):
			if k in SN2ID.keys():
				pIDs[pIDs==k]=SN2ID[k]
			else:
				ind_clear=np.concatenate((ind_clear, np.arange(len(detections))[pIDs==k] ))
		for k,v in SN2ID.items():
			pIDs[pIDs==k]=v 
	else:
		assert set(pIDs).issubset( set(SN2ID.values()) )  
	detections[:,3]=pIDs
	detections=np.delete(detections,ind_clear, axis=0)
	return detections

# function to compile receiver data for all tags into a dictionary
@timing
def compile_tagdata2dict(loc_sys,tagInfo,datafolder,save=True,filename='out_dict.pickle', time_start_UTC=None, time_end_UTC=None):
	# read in all receiver data to a list of dictionaries where each dictionary contains a serial number and combined DataFrame
	data_by_serial_number = process_all_csv_files(datafolder, time_start_UTC=time_start_UTC, time_end_UTC=time_end_UTC)
	tlist = []
	for i, itag in enumerate(tagInfo['code']):
		if itag is None:
			tlist += [{'decodes':None}]
			continue
		logger.info("Tag " +str(itag))
		ndata = get_receiver_data4tag(data_by_serial_number,itag,i)
		if len(ndata)>0:
			cdata = pd.concat(ndata).values
			#tlist += [cdata.copy()]
		else:
			cdata = None
		#
		if cdata is not None:
			cdata=mapNodeCode2ID(loc_sys, cdata)
			tlist += [{'decodes':cdata.astype(np.float64)}]
		else:
			tlist += [{'decodes':None}]
	
	if save == True :
		# Save the dictionary to a pickle file
		with open(filename, 'wb') as f:
			pickle.dump(tlist, f)
		logger.info(f"Dictionary saved to {filename}")


if __name__ == "__main__":
	## data folder for receiver files
	datafolder	= r'MLATS_Datafiles/'
	# file with tag code
	tag_info_file = r'SLAM_auto_locs_w_beacons.csv'
	# column name for Tag Code
	tagcol = 'Beacon'
	#
	compile_tagdata2dict(tag_info_file,datafolder,tagcol,save=True,filename=r'MLATS_Data.pkl')
	# with open('out_dict.pkl', 'rb') as f:
		# loaded_dict = pickle.load(f)
	# print(loaded_dict)
