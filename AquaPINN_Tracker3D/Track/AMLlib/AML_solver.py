import numpy as np
from scipy.linalg import pinv

#speed_of_light = 299792458
#speed_of_sound_underwater = 1482  


def tracking_solver_aml(hydro_loc, tau, ref_rx=0, sound_speed=1500, rangeweight=1, 
                        solved_dimension=2, noises=None, RSR=None):
    """
    Function to solve 2D tracking using acoustic signals.
    
    Parameters:
        hydro_loc (dict): Hydrophone locations dictionary with keys as indices and values as locations.
        tau (float): Time delay.
        sound_speed (float): Speed of sound in the medium.
        rsr (float): Receive Signal Ratio.
        rangeweight (int): Weighting method for range.
        
    Returns:
        src (numpy.ndarray): Estimated source location (x, y).
        jd_best_min (float): Estimated minimum Jaccard distance.
        src_2 (numpy.ndarray): Second estimated source location (x, y).
        jd_best_2_min (float): Second estimated minimum Jaccard distance.
    """
    tau = np.multiply(-1, tau)
    num_iter = 5
    num_sensor = len(hydro_loc)
    N = num_sensor - 1
    origin =  np.array(hydro_loc[ref_rx])  # Assuming the first receiver is the origin
    origin2D =origin[:solved_dimension]
    hydro_loc_array = []
    ref_rx_value = None

    for key, value in hydro_loc.items():
        if key == ref_rx:
            ref_rx_value = value
        else:
            hydro_loc_array.append(value)
    
    # Insert the ref_rx_value at the beginning of the array
    if ref_rx_value is not None:
        hydro_loc_array.insert(0, ref_rx_value)

    hydro_loc_array = np.array(hydro_loc_array)
    hydro_loc_array = hydro_loc_array[:, :solved_dimension]


    hydro_loc_norm = np.linalg.norm(hydro_loc_array[1:], axis=1) ** 2
    D = origin2D - hydro_loc_array[1:]
    delta = np.multiply(sound_speed, tau)
    psi = 2 * delta
    nu1 = delta ** 2 + np.linalg.norm(origin2D) ** 2 - hydro_loc_norm
    JDBest = np.inf * np.ones(6)  # Assuming num_iter = 5
    pos = np.zeros((6, solved_dimension))  # Assuming num_iter = 5

    # weighted LS
    QdInv = np.eye(N) - np.ones((N, N)) / (N + 1)

    if noises is not None:
        noises = np.array(noises)
        QdInv = np.ones((N, N)) + np.diag(1+noises[1:]**2/noises[0]**2)

    enormousFactor = 0.5 * pinv(D.T @ QdInv @ D) @ (D.T @ QdInv)
    c = enormousFactor @ nu1
    b = enormousFactor @ psi
    P = np.linalg.norm(b) ** 2 - 1
    Q = 2 * np.dot(b, c - origin2D)
    R = np.linalg.norm(c - origin2D) ** 2
    if not np.all(np.isfinite([P, Q, R])):
        print('Some intermediate values are infinite')

    r1Roots = np.roots([P, Q, R])
    JDBest[0], pos2D = root_select(b, c, r1Roots, tau, hydro_loc_array, sound_speed, N, QdInv, RSR=RSR)
    pos[0] = pos2D[:solved_dimension]

    # updated weighted LS
    B = np.linalg.norm(pos[0] - hydro_loc_array, axis=1)
    if rangeweight == 1:
        QdInv = np.diag(1 / B[1:]) @ QdInv @ np.diag(1 / B[1:])
    elif rangeweight == 2:
        RR = B[0] ** 2 * np.ones((N, N)) + np.diag(B[1:] * B[1:])
        QdInv = np.linalg.inv(RR)

    enormousFactor = 0.5 * pinv(D.T @ QdInv @ D) @ (D.T @ QdInv)
    c = enormousFactor @ nu1
    b = enormousFactor @ psi
    P = np.linalg.norm(b) ** 2 - 1
    Q = 2 * np.dot(b, c - origin2D)
    R = np.linalg.norm(c - origin2D) ** 2
    if not np.all(np.isfinite([P, Q, R])):
        print('Some intermediate values are infinite')

    r1Roots = np.roots([P, Q, R])
    JDBest[0], pos2D = root_select(b, c, r1Roots, tau, hydro_loc_array, sound_speed, N, QdInv, RSR=RSR)
    pos[0] = pos2D[:solved_dimension]

    # AML updates phi
    for i in range(num_iter):  # Assuming num_iter = 5
        r1 = np.linalg.norm(pos[i] - origin2D)
        r1Term = (pos[i] - origin2D) / r1
        DdDT = np.zeros((num_sensor - 1, solved_dimension))
        lambda_ = np.zeros(num_sensor - 1)
        for k in range(num_sensor - 1):
            rk = np.linalg.norm(pos[i] - hydro_loc_array[k + 1])
            rkTerm = (pos[i] - hydro_loc_array[k + 1]) / rk
            DdDT[k] = rkTerm - r1Term
            lambda_[k] = 1 / (r1 + rk + delta[k])

        W = DdDT.T @ QdInv
        phi = W @ np.diag(lambda_)

        enormousFactor = 0.5 * pinv(phi @ D) @ phi
        c = enormousFactor @ nu1
        b = enormousFactor @ psi
        P = np.linalg.norm(b) ** 2 - 1
        Q = 2 * np.dot(b, c - origin2D)
        R = np.linalg.norm(c - origin2D) ** 2
        if not np.all(np.isfinite([P, Q, R])):
            print('Some intermediate values are infinite')
            break

        r1Roots = np.roots([P, Q, R])
        JDBest[i + 1], pos2D = root_select(b, c, r1Roots, tau, hydro_loc_array, sound_speed, N, QdInv, RSR=RSR)
        pos[i + 1] = pos2D[:solved_dimension]

    index = np.argmin(JDBest)
    src = pos[index]
    T1_travel = np.linalg.norm(src - origin2D) / sound_speed
    RMS_error = np.linalg.norm(src - np.median(hydro_loc_array[:, :solved_dimension]))
    if solved_dimension == 2:
        src = np.append(src, np.nan)
    JDBest_min = JDBest[index]

    return src, JDBest_min



def root_select(b, c, candidates, tau, hydro_loc, sound_speed, N, QdInv, RSR=None):
    if len(candidates) != 2:
        print('Root candidates is not 2!')
    
    if one_is_pos(candidates):
        best_root = max(candidates)
        JDBest, pos = compute_Jd(b, c, best_root, tau, hydro_loc, sound_speed, N, QdInv)
        return JDBest, pos
    
    if np.all(candidates <= 0) or not np.isreal(candidates[0]):
        candidates = np.abs(candidates)
        # You may handle this case differently based on your requirements
        
    # Two positive roots
    Jd1, pos1 = compute_Jd(b, c, candidates[0], tau, hydro_loc, sound_speed, N, QdInv)
    Jd2, pos2 = compute_Jd(b, c, candidates[1], tau, hydro_loc, sound_speed, N, QdInv)
    
    
    if RSR is None:
        if Jd1 <= Jd2:
            pos = pos1
            JDBest = Jd1
        else:
            pos = pos2
            JDBest = Jd2
    elif RSR=='downstreamdam':
        X_ctr=0
        if (pos1[0]-X_ctr)<=0 and (pos2[0]-X_ctr)>=0:
            pos = pos1
            JDBest = Jd1
        
        
        if (pos2[0]-X_ctr)<=0 and (pos1[0]-X_ctr)>=0:
            pos = pos2
            JDBest = Jd2
        
        
        if((pos1[0]-X_ctr)>0 and (pos2[0]-X_ctr)>0):
            pos = np.array([9999,9999,9999])
            JDBest = 9999
    
        
        if (pos1[0]-X_ctr)<=0 and (pos2[0]-X_ctr)<=0:
            if(Jd1 <= Jd2):
                pos = pos1
                JDBest = Jd1
            else:
                pos = pos2
                JDBest = Jd2
            
    elif RSR=='upstreamdam':
        X_ctr=0
        if (pos1[0]-X_ctr)>=0 and (pos2[0]-X_ctr)<=0:
            pos = pos1
            JDBest = Jd1
        
        if (pos2[0]-X_ctr)>=0 and (pos1[0]-X_ctr)<=0:
            pos = pos2
            JDBest = Jd2
        
        if((pos1[0]-X_ctr)<0 and (pos2[0]-X_ctr)<0):
            pos = np.array([9999,9999,9999])
            JDBest = 9999
        
        if (pos1[0]-X_ctr)>=0 and (pos2[0]-X_ctr)>=0:
            if(Jd1 <= Jd2):
                pos = pos1
                JDBest = Jd1
            else:
                pos = pos2
                JDBest = Jd2
    else:
        raise Exception(f"unknown RSR {RSR}")
        
        
        
        
    return JDBest, pos


def compute_Jd(b, c, r1, tau, hydro_loc, sound_speed, N, QdInv):
    pos = (c + b * r1)
    dTheta = np.zeros(np.size(tau))
    for i in range(N):
        dTheta[i] = np.linalg.norm(pos - hydro_loc[i+1])
    dTheta = dTheta - r1
    factor = np.multiply(sound_speed, tau) - dTheta 
    Jd = np.sqrt(np.dot(np.dot(factor, QdInv), factor.T))
    return Jd, pos


def one_is_pos(lst):
    return np.all(np.isreal(lst)) and np.sum(lst > 0) == 1
