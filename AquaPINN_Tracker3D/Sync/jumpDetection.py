# -*- coding: utf-8 -*-


import numpy as np
from scipy import signal
from scipy.stats import skew
def detect_jump(signal_y, threshold,  max_median_ratio=10, filterWidth=7):
    signal_y = signal_y.ravel()
    med_signal_diff = np.percentile(np.abs(np.diff(signal_y)),30)
    signal_y = signal.medfilt(signal_y, filterWidth)
    #smooth
    #signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    
    #signal = np.diff(signal)
    """CUSUM algorithm for detecting change points in a signal."""
    g_pos, g_neg = 0, 0  # Cumulative sums for positive and negative changes
    change_points = []
    pointSet=[]
    flags=[]
    for i in range(1, len(signal_y)):
        delta = signal_y[i] - signal_y[i - 1]  # Calculate difference between points
        pointSet.append(delta)
        g_pos = max(0, g_pos + delta )
        g_neg = min(0, g_neg + delta )
        
        append = False
        if g_pos > threshold:
            change_points.append(i)
            g_pos = 0  # Reset after detecting a change
            append = True
            flags.append(1)
            
        elif g_neg < -threshold:
            change_points.append(i)
            g_neg = 0  # Reset after detecting a change
            append = True
            flags.append(-1)
        if  append:
            arr_pointSet = np.abs(np.array(pointSet))
            max_=np.max(arr_pointSet)
            if max_/(med_signal_diff+1e-14)<max_median_ratio : #
               change_points.pop() 
               flags.pop()
            else:
               pointSet=[]
              
        # if append and len(change_points)>=2:
        #     if (change_points[-1]-change_points[-2])<step_tol:
        #         change_points = change_points[:-2] 
    #change_points = [cp + 1 for cp in change_points]
    return np.array(change_points).astype(int), flags


def jump_fit(signal_x, signal_y, jump_x):
   # matrix
   X = np.column_stack([signal_x, np.ones_like(signal_x)])  # 添加线性项和常数项
   for pos in jump_x:
       X = np.column_stack([X, (signal_x >= pos).astype(float)])  # 添加阶跃项
   # linear regression
   coeffs, _, _, _ = np.linalg.lstsq(X, signal_y, rcond=None)
   # 
   a, b = coeffs[0], coeffs[1]
   jump_y = coeffs[2:]
   # 
   def fitted_function(x):
       # 
       y_fit = a * x + b
       # 
       for pos, coef in zip(jump_x, jump_y):
           y_fit += coef * (x >= pos).astype(float)
       return y_fit
   return fitted_function, (a, b ,jump_y )

def correctJump(ind_jump_x,  jump_y, threshold, max_median_ratio=10, window_size=5):
    iRemove=[]
    #merge jump points
    for i in range(1, len(ind_jump_x)):
        if ind_jump_x[i]-ind_jump_x[i-1]<window_size:
            if np.abs(jump_y[i]+jump_y[i-1])<threshold:
                iRemove += [i-1,i]
            elif jump_y[i]*jump_y[i-1]<=0:
                ind=np.argmin(np.abs(jump_y[i-1:i+1]))
                iRemove.append([i-1,i][ind])
                
    for i in range(0, len(ind_jump_x)):
        if np.abs(jump_y[i])<threshold:
           iRemove.append(i) 
    ind_remove_jump_x = [ind_jump_x[i] for i in iRemove]
    ind_keep_jump_x = list(set(ind_jump_x)-set(ind_remove_jump_x))
    return np.sort(np.array(ind_keep_jump_x).astype(int))
        
def signal_linear_fit(signal_x, signal_y, threshold, window_size=5, max_median_ratio=10, threshold_signalNum=5):
    # Detect change points using CUSUM
    if len(signal_x)<=threshold_signalNum:
        ind_jump_x = np.array([]).astype(int)
        jump_x = 0.5* ( signal_x[ind_jump_x]+signal_x[ind_jump_x-1])
        fitted_function, (a, b ,jump_y) = jump_fit(signal_x, signal_y, jump_x)
        return ind_jump_x, fitted_function, jump_y 
        
    ind_jump_x, flags = detect_jump(signal_y.ravel(), threshold=threshold, max_median_ratio=max_median_ratio)  # Adjust threshold as needed
    for i in range(10):
        jump_x = 0.5* ( signal_x[ind_jump_x]+signal_x[ind_jump_x-1])
        fitted_function, (a, b ,jump_y) = jump_fit(signal_x, signal_y, jump_x)
        ind_keep_jump_x = correctJump(ind_jump_x,  jump_y, threshold=0.8*threshold, window_size=window_size)
        print(f"Iter {i}: {-len(ind_keep_jump_x) + len(ind_jump_x)}/{len(ind_jump_x)} jump points are removed")
        if len(ind_keep_jump_x) == len(ind_jump_x):
            break
        else:
            ind_jump_x=ind_keep_jump_x
    return ind_jump_x, fitted_function, jump_y
    

if __name__=='__main__':
    import matplotlib.pyplot as plt    
    # Example signal
    n_samples = 5
    signal_x = np.linspace(-1, 1-0.01, n_samples)[:, None]
    fun = lambda x: (  10 * (x+np.abs(x) // 0.6*0.2) 
                     + np.random.randn(*x.shape)*0.5
                     + (np.random.rand(*x.shape)>0.8)*4 )
    threshold=1
    signal_y = fun(signal_x)
    
    # signal fit
    ind_jump_x, fitted_function, jump_y = signal_linear_fit(signal_x, signal_y, threshold, window_size=5, max_median_ratio=10)
    jump_x = 0.5* ( signal_x[ind_jump_x]+signal_x[ind_jump_x-1])
    
    pred_y= fitted_function(signal_x)
    residuals = np.abs(signal_y - pred_y)
    inliers = residuals < 1
    inliers_count = np.sum(inliers)    
    inliers_ratio = inliers_count/signal_y.size
    print(f'inliers_ratio={inliers_ratio}')
    
    # Plot the signal and mark the change points
    #plt.close('all')
    plt.figure()
    plt.plot(signal_x, signal_y,'k*')
    #plt.plot(signal_x, np.convolve(signal_y.ravel(), np.ones(window_size)/window_size, mode='same'),'b*')
    for cp in jump_x:
        plt.axvline(x=cp, color='r', linestyle='--')
    plt.plot(signal_x, pred_y, 'g-')
    plt.title('CUSUM Change Point Detection')
    plt.show()




