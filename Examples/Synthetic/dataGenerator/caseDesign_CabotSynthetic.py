# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:52:14 2025
@author: chen096
"""
# %%
import os,copy
import pickle
import numpy as np
from TrajectorySimulator import  ComprehensiveFishSimulator3D
from TOASimulator import loadPhoneLocFile, plotTrack
from TOASimulator import generateTagDecodes,transformTrack
from TOASimulator import generateGPSFile, extractTrack, getTrackCRLB
import matplotlib.pyplot as plt

plt.close('all')
current_dir = os.path.dirname(os.path.abspath(__file__))    

phoneFile=os.path.join(current_dir, 'Cabot_Station_hydrophone_configuration_update_3.csv')
phoneInfo = loadPhoneLocFile(phoneFile)
phoneLoc  = phoneInfo['XYZ']
ENE0 = phoneInfo['ENE0']
theta = phoneInfo['theta']

region_lb=(-3000,-3000,-10)
region_ub=(3000,3000,0)
total_time=600 # sec
dt=0.1
accel=np.array([0.5,0.5,0.01])*2
init_speed=0.5
fout=None
noise = 3E-4
PRI = 2


# Critical tests
outputFolder= os.path.join(current_dir, '..', 'data','Critical' )
Scenarios ={i:None for i in [1,4,5,9]} #[1, 2, 3, 4, 5, 6, 7, 8]}
draw_tracks = True


# CRLB with varying noise
# outputFolder= os.path.join(current_dir, '..', 'data','CRLB' )
# Scenarios ={i:None for i in [11]} 
# draw_tracks = False

# # senstivity analysis
# outputFolder= os.path.join(current_dir, '..', 'data','Sensitivity' )
# Scenarios ={i:None for i in [1, 111]} #[1, 2, 3, 4, 5, 6, 7, 8]}
# draw_tracks = False

os.makedirs(outputFolder, exist_ok=True)

# %%
sim = ComprehensiveFishSimulator3D(
    dt=dt,
    total_time=total_time,
    random_seed=42,
    region_lb=region_lb,
    region_ub=region_ub,
)
t = sim.time_array
fp = 5*np.pi/total_time
px = -50*np.sin(0.6*t/total_time)
py = 40*np.sin(0.8*np.cos(fp*t) )
pz = -2+1*np.cos(2*np.pi*t/total_time)
p  = np.stack((px,py,pz), axis=1)


    


# %%
# Scenario 1: baseline setup
if True:
    Scenarios[1] = modes =['baseline']
    mode = modes[0]
    track0 = sim.generate_base_trajectory(
        basePositions=p,
        accel =accel,
    )
    vmag = np.linalg.norm(track0['vel'], axis=1)
    print(f"velocity: median={np.median(vmag)}, mean={np.mean(vmag)}, max={np.max(vmag)}")    
    plotTrack(track0, phoneLoc, title='baseline', fout=fout)
    track0 = transformTrack(track0, np.array([[-25,0,0]]), theta=-0)
    track0['XYZ'] +=np.array([[20,-10,0]])
    track1 = extractTrack(track0, PRI)        
    plotTrack(track1, phoneLoc, title=mode.replace('_', ' '), fout=os.path.join(outputFolder,f'{mode}.jpg'))
    tagDecodes = generateTagDecodes(track1, phoneLoc, noise=noise, fout=os.path.join(outputFolder,f'{mode}.pickle'))
    generateGPSFile(track1, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),
                    ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta'])        
    modes=[]
    if 11 in Scenarios:
        for noise_i in [1E-4, 2E-4, 4E-4, 8E-4, 1.6E-3,3.2E-3,6.4E-3]:
            for i in range(101):
                for allowMissing in [False, True]:
                    mode = f'baseline_{"missing" if allowMissing else "nomissing"}_noise{noise_i}_{i}'
                    modes += [mode]
                    tagDecodes = generateTagDecodes(track1, phoneLoc, noise=noise_i, fout=os.path.join(outputFolder,f'{mode}.pickle'),
                                                    applyMissing=allowMissing)    
                    track11 = getTrackCRLB(track1, phoneLoc, noise=noise_i)
                    generateGPSFile(track11, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),
                                   ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta'])
        Scenarios[11]=modes  
    if 111 in Scenarios: # add outliers for baseline
        mode = f'baseline_outliers'
        Scenarios[111]=[mode]  
        tagDecodes = generateTagDecodes(track1, phoneLoc, noise=noise, fout=os.path.join(outputFolder,f'{mode}.pickle'), noise_outlier=0.1, outlierRatio=2)
        generateGPSFile(track1, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta'])        

# Scenario 2: Varying speed
if 2 in Scenarios:
    Scenarios[2]=modes = ['2x_velocity', '4x_velocity']
    for scale, mode in zip([2,4], modes):
        ps = p*scale
        track20 = sim.generate_base_trajectory(
            basePositions=ps,
            accel =accel*scale,
        )
        vmag = np.linalg.norm(track20['vel'], axis=1)
        print(f"velocity: median={np.median(vmag)}, mean={np.mean(vmag)}, max={np.max(vmag)}")
        track2 = transformTrack(track20, np.array([[-25,0,0]]), theta=-0)
        track2['XYZ'] +=np.array([[20*scale,-10,0]])
        track2 = extractTrack(track2, PRI)        
        plotTrack(track2, phoneLoc, title=mode.replace('_', ' '), fout=os.path.join(outputFolder,f'{mode}.jpg'))
        tagDecodes = generateTagDecodes(track2, phoneLoc, noise=noise, fout=os.path.join(outputFolder,f'{mode}.pickle'))
        generateGPSFile(track2, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),
                        ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta'])    
# Scenario 3: Varying noise        
if 3 in Scenarios:
    Scenarios[3]=modes = ['2x_noise', '4x_noise']
    for scale,mode in zip([2,4], modes):
        tagDecodes = generateTagDecodes(track1, phoneLoc, noise=noise, fout=os.path.join(outputFolder,f'{mode}.pickle'))
        generateGPSFile(track1, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),
                        ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta']) 

# Scenario 4: sudden turn
if 4 in Scenarios:
    Scenarios[4] = modes =['sudden_turn']
    mode = modes[0]
    track4 = sim.apply_sudden_turn(track0,
                                    ts_turn=[total_time/2],
                                    angles_turn=[np.pi/2],)    
    vmag = np.linalg.norm(track4['vel'], axis=1)
    print(f"velocity: median={np.median(vmag)}, mean={np.mean(vmag)}, max={np.max(vmag)}")
    track4['XYZ'] +=np.array([[-10,5,0]])
    track4 = extractTrack(track4, PRI)    
    plotTrack(track4, phoneLoc, title=mode.replace('_', ' '), fout=os.path.join(outputFolder,f'{mode}.jpg'))
    tagDecodes = generateTagDecodes(track4, phoneLoc, noise=noise, fout=os.path.join(outputFolder,f'{mode}.pickle'))
    generateGPSFile(track4, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),
                    ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta']) 
# Scenario 5: sudden accerlation
if 5 in Scenarios:
    Scenarios[5] = modes =['sudden_burst']
    mode = modes[0]
    track5 = sim.apply_sudden_burst(track0,
                                    t_interval=[total_time*0.3, total_time*0.4],
                                    burst_factor=4,)  
    vmag = np.linalg.norm(track5['vel'], axis=1)
    print(f"velocity: median={np.median(vmag)}, mean={np.mean(vmag)}, max={np.max(vmag)}")
    track5['XYZ'] +=np.array([[0,-35,0]])
    track5 = extractTrack(track5, PRI)
    
    plotTrack(track5, phoneLoc, title=mode.replace('_', ' '), fout=os.path.join(outputFolder,f'{mode}.jpg'))
    tagDecodes = generateTagDecodes(track5, phoneLoc, noise=noise, fout=os.path.join(outputFolder,f'{mode}.pickle'))
    generateGPSFile(track5, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),
                    ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta'])     


# Scenario 6: short term
if 6 in Scenarios:
    Scenarios[6]=modes = ['short_term_1min']
    mode = modes[0]
    t = track1['t']
    ind  = t<120
    track6={'t':track1['t'][ind], 'XYZ':track1['XYZ'][ind], 'vel':track1['vel'][ind]}
    track6 = extractTrack(track6, PRI)
    plotTrack(track6, phoneLoc, title=mode.replace('_', ' '), fout=os.path.join(outputFolder,f'{mode}.jpg'))
    tagDecodes = generateTagDecodes(track6, phoneLoc, noise=noise, fout=os.path.join(outputFolder,f'{mode}.pickle'))
    generateGPSFile(track6, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),
                    ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta']) 

# Scenario 7: missing gap
if 7 in Scenarios:
    Scenarios[7]=modes = ['gap_3hour']
    mode = modes[0]
    t = track1['t']    
    track7 = copy.deepcopy(track1)
    ind  = (t>400) | (t<200)
    track7['t'][t>300] += 3600*3    
    track7={'t':track7['t'][ind], 'XYZ':track7['XYZ'][ind], 'vel':track7['vel'][ind]}    
    plotTrack(track7, phoneLoc, title=mode.replace('_', ' '), fout=os.path.join(outputFolder,f'{mode}.jpg'))
    tagDecodes = generateTagDecodes(track7, phoneLoc, noise=noise, fout=os.path.join(outputFolder,f'{mode}.pickle'))
    generateGPSFile(track7, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),
                    ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta']) 

# Scenario 8: varying distance
if 8 in Scenarios:
    Scenarios[8]=modes = ['distanceV1', 'distanceV2', 'distanceH1', 'distanceH2', ]
    dxy=[[40,0],[-40,0],[0,40],[0,-40]]
    for (dx,dy), mode in zip(dxy, modes):
        track8 = copy.deepcopy(track0)
        track8['XYZ'] +=np.array([[dx,dy,0]])
        track8 = extractTrack(track8, PRI)        
        plotTrack(track8, phoneLoc, title=mode.replace('_', ' '), fout=os.path.join(outputFolder,f'{mode}.jpg'))
        tagDecodes = generateTagDecodes(track8, phoneLoc, noise=noise, fout=os.path.join(outputFolder,f'{mode}.pickle'))
        generateGPSFile(track8, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),
                        ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta'])    

if 9 in Scenarios:
    Scenarios[9]=modes = ['tmp_Exit' ]
    mode = modes[0]
    track9 = sim.apply_sudden_burst(track0,
                                    t_interval=[total_time*0.2, total_time*0.6],
                                    burst_factor=5, time_factor=5)  
    vmag = np.linalg.norm(track9['vel'], axis=1)
    print(f"velocity: median={np.median(vmag)}, mean={np.mean(vmag)}, max={np.max(vmag)}")
    track9['XYZ'][:,0] *=0.5
    track9['XYZ'] +=np.array([[10,-20,0]])
    track9 = extractTrack(track9, PRI)
    
    
    plotTrack(track9, phoneLoc, title=mode.replace('_', ' '), fout=os.path.join(outputFolder,f'{mode}.jpg'))
    tagDecodes = generateTagDecodes(track9, phoneLoc, noise=noise, fout=os.path.join(outputFolder,f'{mode}.pickle'))
    generateGPSFile(track9, fout=os.path.join(outputFolder,f'{mode}_GPS.csv'),
                    ENE0=phoneInfo['ENE0'], theta=phoneInfo['theta'])     

with open(os.path.join(outputFolder,'Scenarios.pickle'), 'wb') as f:
    pickle.dump(Scenarios, f)    
    
if draw_tracks:    
    # draw track1, track4, track5, track9 into four subplots, show phone locations as red pentagons with labels
    fontsize =16
    titles =['(a)', '(b)', '(c)', '(d)']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (track, mode) in zip(axs.ravel(), [(track1, 'baseline'), (track4, 'sudden_turn'), (track5, 'sudden_burst'), (track9, 'temporarily exit')]):
        ind_track = axs.ravel().tolist().index(ax)
        ax.plot(track['XYZ'][:, 0], track['XYZ'][:, 1], '-ro', markersize=2)
        # plot phone locations
        ax.plot(phoneLoc[:, 0], phoneLoc[:, 1], 'p', color='red', markersize=10, label='Receivers')
        # label each phone
        for idx, (x, y) in enumerate(phoneLoc[:, :2], 1):
            ax.text(x, y, str(idx), color='black', fontsize=10, ha='center', va='center')
        # ax.set_title(mode.replace('_', ' '), fontsize=fontsize)
        title = f"{titles[ind_track]} {mode.replace('_', ' ')}"
        # ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(f'X (m)\n {title}', fontsize=fontsize)
        ax.set_ylabel('Y (m)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.legend(fontsize=fontsize-2)
        #ax.axis('equal')
    plt.tight_layout()

    plt.savefig(os.path.join(outputFolder, 'Scenarios_tracks.jpg'), dpi=300, bbox_inches='tight')
