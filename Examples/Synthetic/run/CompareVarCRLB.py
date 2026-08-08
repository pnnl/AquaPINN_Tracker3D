import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

dataFolder = os.path.join(SCRIPT_DIR, '..','data', 'CRLB')    
resultFolder_root = os.path.join(SCRIPT_DIR, 'results','CRLB') 


ScenarioFile = os.path.join(dataFolder, 'Scenarios.pickle')
k=-1
noise_Vec = [1E-4, 2E-4, 4E-4, 8E-4, 1.6E-3, 3.2E-3, 6.4E-3][:]
methods = ['NN', 'NN_noMissing','NNplus', 'NNplus_noMissing', ]
err2 = np.zeros((101,len(noise_Vec),3))
Num  = np.zeros((101,len(noise_Vec),3))
crlb = np.zeros((len(noise_Vec),3))
err2_plus= np.zeros((101,len(noise_Vec),3))
Num_plus  = np.zeros((101,len(noise_Vec),3))
eff       = np.zeros((len(noise_Vec)))
eff_plus       = np.zeros((len(noise_Vec)))
err_all={method:None for method in methods}
eff_all={method:None for method in methods}
for method in methods:
    is_noMissing = 'noMissing' in method
    methodName   = method.split('_')[0]
    for id_noise,noise in enumerate(noise_Vec):   
        mode=f'baseline_{"nomissing" if is_noMissing else "missing" }_noise{noise}_{0}'    
        GPSFile = os.path.join(dataFolder, f'{mode}_GPS.csv')
        assert os.path.exists(GPSFile), f"GPS file {GPSFile} does not exist."
        GPS = pd.read_csv(GPSFile)
        crlb[id_noise] = np.sqrt(np.square(GPS[['CRLB_X', 'CRLB_Y', 'CRLB_Z']]).mean(axis=0))  
        for repeatID in range(101):
            mode = f'baseline_{"nomissing" if is_noMissing else "missing" }_noise{noise}_{repeatID}'   
            trackFile=os.path.join(resultFolder_root, mode, "Track", 'track',f'track_Tag-1_SYNTHETIC_TAG_{methodName}.csv')
            assert os.path.exists(trackFile), f"Track file {trackFile} does not exist."
            def getErrorNum(trackFile):
                track = pd.read_csv(trackFile)
                error = track[['X error', 'Y error', 'Z error']].values
                std   = track[['X std', 'Y std', 'Z std']].values
                ind_std = np.linalg.norm(std[:,:3], axis=1) <5
                ind = np.all(np.isfinite(error), axis=1)   
                error = error[ind&ind_std, :]
                if np.sum(~ind)>0:
                    print(f"Warning: {np.sum(~ind)} NaN values found in the error data for noise level {noise} and repeat ID {repeatID}.")
                if np.sum(~ind_std)>0:
                    print(f"Warning: {np.sum(~ind_std)} std values below threshold for noise level {noise} and repeat ID {repeatID}.")
                return np.sum(error**2, axis=0), len(error)
            err2[repeatID, id_noise], Num[repeatID, id_noise] = getErrorNum(trackFile)
    err_all[method] = np.sqrt(np.sum(err2, axis=0) / np.sum(Num, axis=0))
    eff_all[method] = np.mean(Num, axis=0)[:,0]/301


noise_Vec = np.array(noise_Vec) * 10000 

fontsize = 18
# plt in 3 subfigures
dims = ['X', 'Y', 'Z']
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plot X, Y, Z RMS Error in first three subplots
listIndex=['(a)', '(b)', '(c)', '(d)']
for i in range(3):
    ax = axs[i // 2, i % 2]
    for method  in err_all.keys():
        err = err_all[method]
        ax.plot(noise_Vec, err[:, i], marker='o', linestyle='-', linewidth=2, label=f'{method}'.replace('plus','+').replace('_noMissing',', No missing'))
    ax.plot(noise_Vec, crlb[:, i], marker='x', linestyle='--', linewidth=2, label='CRLB')
    ax.set_xlabel(f'{listIndex[i]} Noise Std. Dev. ($10^{-4}$ sec)', fontsize=fontsize)
    ax.set_ylabel(f'{dims[i]} RMS Error (m)', fontsize=fontsize)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(noise_Vec)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
    ax.legend(fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

# Plot efficiency in the fourth subplot
ax_eff = axs[1, 1]
for method  in err_all.keys():
    eff = eff_all[method]
    ax_eff.plot(noise_Vec, eff, marker='o', linestyle='-', linewidth=2, label=f'{method}'.replace('plus','+').replace('_noMissing',', No missing'))
ax_eff.set_xlabel(f'{listIndex[3]} Noise Std. Dev. ($10^{-4}$ sec)', fontsize=fontsize)
ax_eff.set_ylabel('Tracking efficiency', fontsize=fontsize)
ax_eff.set_xscale('log', base=2)
ax_eff.set_xticks(noise_Vec)
ax_eff.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax_eff.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
ax_eff.legend(fontsize=fontsize)
ax_eff.tick_params(axis='both', which='major', labelsize=fontsize)

fig.tight_layout()
plt.savefig(os.path.join(resultFolder_root, '..', 'CRLB_vs_NN_RMS_Error_and_Efficiency.png'), dpi=300, bbox_inches='tight')
plt.show()


