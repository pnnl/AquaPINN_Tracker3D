# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:22:57 2024

@author: chen096
"""
import numpy as np
import pandas as pd
from .ML_NN import soundSpeed_freshWater, get_PRI
from .AMLlib import AML_solver_w_rx_selection_pd
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)

def AMLSolver(data, region_lb, region_ub, ignoreZ,
            temperature_ref, dis_detect, RSR=None):
    dis_tol=3
    dis_limit = 250
    first_ref_flag = 1

    RSR = RSR.lower() if isinstance(RSR, str) else None    
        
    t, toa_noisy, phoneLoc, phoneIDs, mask, TOA_t_num=data
    soundSpeed_ref = soundSpeed_freshWater(temperature_ref)
    phoneIDs = np.array(phoneIDs).astype(int)
    dim =2 if ignoreZ else 3
    ind, xyz=[], []
    Nt = len(t)
    bounds = np.stack( (region_lb,region_ub), 
                             axis=0).squeeze().T
    if RSR == 'downstreamdam':  
        bounds[0,1] =0
    elif RSR == 'upstreamdam':
        bounds[0,0] =0
    else:
        RSR=None
    nBC_filtered=0
    nDiff_filtered=0
    nDis_filtered=0
    nFail=0
    for i in tqdm(range(Nt), desc='AML Solving'):
        TOAs = toa_noisy[i]
        maski = mask[i]
        if np.sum(maski) < dim+1 :
            continue
        

        tol_filter = 0
        cur_toas = TOAs[maski]
        cur_rx_pos = phoneLoc[maski]  
        cur_rx_ids  = phoneIDs[maski]

        
        while tol_filter == 0:
            pos_rx = dict( zip(cur_rx_ids, cur_rx_pos) )
            sensor_cnt = len(cur_rx_ids)
            if (tol_filter == 0 and sensor_cnt >= dim + 1):

                # Construct a TDOA array
                tdoa_array = {'rx0': [], 'rx1': [], 'tdoa': []}
                for ik in range(len(cur_rx_ids) - 1):
                    for jk in range(ik + 1, len(cur_rx_ids)):
                        tdoa_array['rx0'].append(cur_rx_ids[ik])
                        tdoa_array['rx1'].append(cur_rx_ids[jk])
                        tdoa_array['tdoa'].append((cur_toas[ik] - cur_toas[jk])) #??

                tdoa_array['tdoa'] = np.array(tdoa_array['tdoa'])
                tdoa_array['rx0'] = np.array(tdoa_array['rx0'])
                tdoa_array['rx1'] = np.array(tdoa_array['rx1'])

                test_group = AML_solver_w_rx_selection_pd.TDOAGroup(pd.DataFrame(tdoa_array), pos_rx,
                                                                    sound_speed=soundSpeed_ref,
                                                                    noises=None, RSR=RSR)
                try:
                    if first_ref_flag:
                        combineData = np.column_stack((cur_toas, cur_rx_ids))
                        sorted_indices = np.argsort(combineData[:, 0])
                        combineData = combineData[sorted_indices]
                        ref_rx = combineData[0, 1]
                        xyzi, _ , _ = test_group.solve_w_ref_receiver_internal(ref_rx, "tdoa", dim)
                    else:
                        xyzi, ref_rx = test_group.solve_w_best_receiver("tdoa", dim)
                    xyzi[dim:] = 0
                    if np.any(xyzi[:dim]<=bounds[:dim,0] ) or np.any(xyzi[:dim]>=bounds[:dim,1] ):
                        if sensor_cnt <= dim + 1:
                            tol_filter = 1
                        else:
                            remove_idx = np.argmax(cur_toas)
                            remove_idx_ori = np.where(maski)[0][remove_idx]
                            cur_toas = np.delete(cur_toas, remove_idx)
                            cur_rx_pos = np.delete(cur_rx_pos, remove_idx, axis=0)
                            cur_rx_ids = np.delete(cur_rx_ids, remove_idx)
                            mask[i][remove_idx_ori] = False
                            nBC_filtered+=1
                    else:
                        dis = np.linalg.norm(xyzi-cur_rx_pos, axis=1)
                        ref_idx = np.where(cur_rx_ids == ref_rx)[0][0]
                        diff = abs((cur_toas - cur_toas[ref_idx]) * soundSpeed_ref - (dis - dis[ref_idx]))
                        if dis_tol is not None and np.any(diff > dis_tol):
                            if sensor_cnt <= dim + 1:
                                tol_filter = 1
                            else:
                                remove_idx = np.argmax(diff)
                                remove_idx_ori = np.where(maski)[0][remove_idx]
                                cur_toas = np.delete(cur_toas, remove_idx)
                                cur_rx_pos = np.delete(cur_rx_pos, remove_idx, axis=0)
                                cur_rx_ids = np.delete(cur_rx_ids, remove_idx)
                                mask[i][remove_idx_ori] = False
                                nDiff_filtered+=1
                        else:
                            if dis_limit is not None and np.any(dis > dis_limit):
                                if sensor_cnt <= dim + 1:
                                    tol_filter = 1
                                else:
                                    remove_idx = np.argmax(dis)
                                    remove_idx_ori = np.where(maski)[0][remove_idx]
                                    cur_toas = np.delete(cur_toas, remove_idx)
                                    cur_rx_pos = np.delete(cur_rx_pos, remove_idx, axis=0)
                                    cur_rx_ids = np.delete(cur_rx_ids, remove_idx)
                                    mask[i][remove_idx_ori] = False
                                    nDis_filtered+=1
                            else:
                                tol_filter = 1
                                ind.append(i)
                                xyz.append(xyzi)                               
                except ValueError:
                    tol_filter = 1
                    nFail+=1                
    if len(xyz)==0:
        return None
    ind  = np.array(ind).astype(int)
    xyz = np.stack(xyz, axis=0)
    from scipy import signal 
    for i in range(dim):
        xyz[:,i] = signal.medfilt(xyz[:,i], 5)
    nSolved   = xyz.shape[0] 
    nSolvable = np.sum(np.sum(mask, axis=1)>=dim+1)
    solve_eff = nSolved / nSolvable
    logger.info(f'{nFail} points failed in resolving')
    logger.info(f'Total {nSolvable} solvable points')
    logger.info(f'Filtered {nBC_filtered} points out of boundary')
    logger.info(f'Filtered {nDiff_filtered} points with large TDOA residuals')
    logger.info(f'Filtered {nDis_filtered} points beyond distance limit')    
    logger.info(f'AML solver solving efficieny: {nSolved}/{nSolvable}={solve_eff*100:.2f}%')
    
    dt=np.diff(TOA_t_num)*24*3600
    t_total = np.sum( dt[dt<300] )      
    PRI = get_PRI(toa_noisy, mask)
    efficiency=xyz.shape[0] / (t_total/PRI)
    timestamps = pd.to_datetime( TOA_t_num[ind], unit='D').round('s')
    track_AML={'t_datetime':timestamps,
            't_num': np.array([ ti.timestamp() for ti in timestamps]),
            'time':t[ind],
            'XYZ': xyz,
            'XYZ_all': xyz[None,...],    
            'XYZ_std': np.zeros_like(xyz),            
            'phoneLoc':phoneLoc,
            'phoneIDs':phoneIDs,
            'toa_noisy':toa_noisy[ind],
            'weight':np.ones_like(toa_noisy[ind]),
            'top': np.zeros_like(toa_noisy[ind,0]), # dummy top for compatibility with NN tracker
            'toa_tau':np.ones_like(toa_noisy[ind]),
            'soundSpeed':soundSpeed_ref,
            'efficiency':efficiency,
            'nValid': np.sum(mask[ind], axis=1),
            'PRI': PRI,
            'dim':dim,
            }                   
    return  track_AML
