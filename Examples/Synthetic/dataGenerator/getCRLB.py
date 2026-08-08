import numpy as np

def compute_fim_toa_3d(theta, hydro_pos, sigma_t, c=1500):
    """
    Compute the 3×3 Fisher Information Matrix (FIM) for a 3D TOA system.
    
    Parameters
    ----------
    theta : array_like, shape (3,)
        [x, y, z] coordinates of the source.
    hydro_pos : array_like, shape (N,3)
        Positions of the N hydrophones.
    sigma_t : float
        Standard deviation of TOA noise (seconds).
    c : float
        Speed of sound in water (m/s). Default is 1500.
        
    Returns
    -------
    J : ndarray, shape (3,3)
        Fisher Information Matrix at theta.
    """
    J = np.zeros((3, 3))
    for p in hydro_pos:
        diff = theta - p
        d = np.linalg.norm(diff)
        grad = diff / (c * d)         # ∇_θ τ_i (3×1)
        J += np.outer(grad, grad)     # accumulate outer product
    J /= sigma_t**2
    return J

def compute_crlb_toa_3d(theta, hydro_pos, sigma_t, c=1500):
    """
    Compute the 3×3 CRLB covariance matrix for a 3D TOA system.
    Returns the inverse of the FIM.
    """
    J = compute_fim_toa_3d(theta, hydro_pos, sigma_t, c)
    return np.diag(np.linalg.inv(J))

# Example usage:
if __name__ == "__main__":
    # 3D hydrophone geometry (meters)
    hydro_pos = np.array([
        [ 100.0,   0.0,   0.0],
        [   0.0, 100.0,   0.0],
        [-100.0,   0.0,   0.0],
        [   0.0,-100.0,   0.0],
        [   0.0,   0.0, 1.0],
        [   0.0,   0.0,-1.0],
    ])
    
    # Noise and sound speed
    sigma_t = 3e-4   # 0.1 ms timing jitter
    c       = 1500   # m/s

    # Point at which to compute CRLB
    theta = np.array([10, 0.5, 2])  

    # Compute CRLB covariance
    C = compute_crlb_toa_3d(theta, hydro_pos, sigma_t, c)

    # Extract RMSE bounds
    rmse_x    = np.sqrt(C[0])
    rmse_y    = np.sqrt(C[1])
    rmse_z    = np.sqrt(C[2])
    rmse_tot  = np.sqrt(np.sum(C))

    print("CRLB covariance matrix (m²):\n", C)
    print(f"RMSE_x     ≥ {rmse_x:.4f} m")
    print(f"RMSE_y     ≥ {rmse_y:.4f} m")
    print(f"RMSE_z     ≥ {rmse_z:.4f} m")
    print(f"Total RMSE ≥ {rmse_tot:.4f} m")
