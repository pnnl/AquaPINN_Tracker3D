import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotate_vector_xy(v, angle):
    """
    Rotate the 3D vector v by an angle in x-y plane.
    """
    cos=np.cos(angle)
    sin=np.sin(angle)
    vnew = v*0
    vnew[...,0]=v[...,0]*cos - v[...,1]*sin 
    vnew[...,1]=v[...,0]*sin + v[...,1]*cos
    vnew[...,2]=v[...,2]
    return vnew


class ComprehensiveFishSimulator3D:
    """
    A comprehensive 3D fish movement simulator that incorporates:
      1. Reflective boundary region
      2. Layered environment (z_split)
      3. Correlated random walk with per-axis variance (anisotropy)
      4. Limiting vertical angle (max_z_ratio)
      5. Sudden events: turn, accel, or both
      6. Speed modes: slow/burst
      7. Explicit initial velocity (init_vel)
    """
    def __init__(
        self,
        dt=0.1,
        total_time=60.0,
        init_pos=(0,0,-2),
        init_vel=None,      # If None, default is (init_speed, 0, 0)
        init_speed=0.5,
        random_seed=42,
        region_lb=(-10,-10,-10),
        region_ub=(10,10,0)
    ):
        """
        :param dt: time step
        :param total_time: total simulation duration
        :param init_pos: initial position (x, y, z)
        :param init_vel: initial velocity vector (vx, vy, vz).
                         If provided, overrides init_speed.
        :param init_speed: if init_vel is only a direction or None, use this speed
        :param random_seed: for reproducibility
        :param region_lb: (x_min, y_min, z_min)
        :param region_ub: (x_max, y_max, z_max)
        """
        np.random.seed(random_seed)
        self.dt = dt
        self.total_time = total_time
        self.time_array = np.arange(0, total_time, dt)
        self.num_steps = len(self.time_array)

        self.init_pos = np.array(init_pos, dtype=float)

        # Handle init_vel logic
        if init_vel is None:
            # If not specified, default to (init_speed, 0, 0)
            self.init_vel = np.array([init_speed, 0.0, 0.0])
        else:
            vel_array = np.array(init_vel, dtype=float)
            norm_v = np.linalg.norm(vel_array)
            if norm_v < 1e-12:
                # If user gave (0,0,0), fallback to (init_speed,0,0)
                self.init_vel = np.array([init_speed, 0, 0])
            else:
                # If norm_v ~ 1, treat it as a direction; otherwise it's a full velocity
                if abs(norm_v - 1.0) < 1e-6:
                    self.init_vel = vel_array / norm_v * init_speed
                else:
                    self.init_vel = vel_array

        # Bounding region
        self.region_lb = np.array(region_lb, dtype=float)
        self.region_ub = np.array(region_ub, dtype=float)


    def reflect_if_outside(self, pos, vel):
        """
        Reflective boundary: if any coordinate goes outside region_lb/region_ub,
        reflect the position about that boundary and invert the velocity in that dimension.
        """
        for dim in range(3):
            if pos[dim] < self.region_lb[dim]:
                pos[dim] = 2*self.region_lb[dim] - pos[dim]
                vel[dim] = -np.random.rand()*vel[dim]
            elif pos[dim] > self.region_ub[dim]:
                pos[dim] = 2*self.region_ub[dim] - pos[dim]
                vel[dim] = -np.random.rand()*vel[dim]
        return pos, vel

    def generate_base_trajectory0(
        self,
        sigma_x=0.2,
        sigma_y=0.2,
        sigma_z=0.05,
    ):
        """
        Generate the initial correlated random walk trajectory (anisotropic),
        applying reflection at boundaries and limiting vertical angle.
        Does not include sudden events or speed profiles yet.
        """
        positions = np.zeros((self.num_steps, 3)) 
        velocities = np.zeros((self.num_steps, 3))

        # Initialize
        positions[0] = self.init_pos
        velocities[0] = self.init_vel

        for i in range(1, self.num_steps):
            prev_v = velocities[i-1]
            # Create anisotropic random vector
            rx = np.random.normal(0, sigma_x)
            ry = np.random.normal(0, sigma_y)
            rz = np.random.normal(0, sigma_z)
            rand_ani = np.array([rx, ry, rz])

            new_v = prev_v + rand_ani

            # Update position with reflection check
            new_pos = positions[i-1] + new_v*self.dt
            new_pos, new_v_ref = self.reflect_if_outside(new_pos, new_v)

            positions[i] = new_pos
            velocities[i] = new_v_ref

        return self.time_array, positions, velocities

    def generate_base_trajectory(
        self,
        basePositions,
        accel=[0.2, 0.2,0.05],        
    ):
        
        vel = np.diff(basePositions, axis=0)/self.dt
        vel = np.concatenate((vel[:1,:],vel), axis=0)
    
        vel = vel + np.random.normal(size=vel.shape)*np.array([accel])*self.dt
        
        new_pos = self.recompute_positions(basePositions[0], vel)
        
        track ={'t':self.time_array, 'XYZ':new_pos, 'vel':vel}
        return track
    
    def apply_sudden_turn(self, track, ts_turn, angles_turn, ):
        times, positions, velocities = track['t'], track['XYZ'], track['vel']
        new_vel =velocities+0
        for t_turn, angle_turn in zip(ts_turn, angles_turn):
            ind = np.where(times>t_turn)[0][0]
            new_vel=new_vel+0
            new_vel[ind:] = rotate_vector_xy(new_vel[ind:], angle_turn)
            new_pos = self.recompute_positions(positions[0], new_vel)
        track ={'t':times, 'XYZ':new_pos, 'vel':new_vel}
        return track
    def apply_sudden_burst(self, track, t_interval, burst_factor, time_factor=1):
        times, positions, velocities = track['t'], track['XYZ'], track['vel']
        t_st,t_nd=t_interval
        ind_st =  np.where(times>t_st)[0][0]
        ind_nd =  np.where(times>t_nd)[0][0]
        t_norm = (times[ind_st:ind_nd]-(times[ind_st]+times[ind_nd])/2)/(times[ind_st]-times[ind_nd])*2
        new_vel = velocities + 0
        new_vel[ind_st:ind_nd] *= burst_factor*(1-t_norm[:,None]**8)
        new_pos = self.recompute_positions(positions[0], new_vel)
        

        new2_pos_st = new_pos[:ind_st]
        new_times_st = times[:ind_st]
        ind_mid = np.linspace(ind_st, ind_nd, int((ind_nd-ind_st)*time_factor)+1)
        new_times_mid = times[ind_nd] + (ind_mid-ind_mid[0])/(ind_mid[-1]-ind_mid[0])*(times[ind_nd]-times[ind_st])*time_factor 
        new2_pos_mid = np.zeros((len(ind_mid),3))
        # interpolate from new_pos to new2_pos
        for dim in range(3):
            new2_pos_mid[:,dim] = np.interp(ind_mid, np.arange(ind_st,ind_nd+1), new_pos[ind_st:ind_nd+1,dim])
        new2_pos_nd = new_pos[ind_nd:]
        new_times_nd = times[ind_nd:]-times[ind_nd]+new_times_mid[-1]
        times = np.concatenate((new_times_st, new_times_mid, new_times_nd), axis=0)
        new_pos = np.concatenate((new2_pos_st, new2_pos_mid, new2_pos_nd), axis=0)
        new_vel = np.gradient(new_pos, axis=0)/np.gradient(times)[:,None]
        track ={'t':times, 'XYZ':new_pos, 'vel':new_vel}
        return track
    
    def recompute_positions(self, init_pos, velocities):
        """
        After final modifications to velocities (e.g. sudden events, speed mode),
        integrate them to get positions, applying reflection at each step.
        """
        pos = np.zeros_like(velocities)
        pos[0] = init_pos
        for i in range(1, self.num_steps):
            candidate = pos[i-1] + velocities[i]*self.dt
            candidate, new_v = self.reflect_if_outside(candidate, velocities[i])
            pos[i] = candidate
            velocities[i] = new_v
        return pos 
 
if __name__=="__main__":
    region_lb=(-3000,-3000,-10)
    region_ub=(3000,3000,0)
    total_time=600
    dt=0.1
    accel=np.array([0.5,0.5,0.01])*10
    prob_event = 1E-3
    # Scenario A: no layering, no sudden events, just random walk + reflection
    simA = ComprehensiveFishSimulator3D(
        dt=dt,
        total_time=total_time,
        init_pos=(0,0,-2),
        init_vel=(1, 1, 0),  # explicit initial velocity
        init_speed=0.5,      # used only if init_vel is a direction
        random_seed=42,
        region_lb=region_lb,
        region_ub=region_ub,
    )
    t = simA.time_array
    fp = 5*np.pi/total_time
    px = 0.5/fp*np.exp(-t/total_time)*np.sin(fp*t) + 0.2*t
    py = 0.5/fp*np.exp(-t/total_time)*np.cos(fp*t) 
    pz = -2+1*np.cos(2*np.pi*t/total_time)
    p  = np.stack((px,py,pz), axis=1)
    trackA = simA.generate_base_trajectory(
        basePositions=p,
        accel =accel,
    )
    posA = trackA['XYZ']
    trackB = simA.apply_sudden_turn(trackA,
                                    ts_turn=[total_time/3, total_time*2/3],
                                    angles_turn=[np.pi*3/4, np.pi/2],)
    posB = trackB['XYZ']
    trackC = simA.apply_sudden_burst(trackA,
                                    t_interval=[total_time*0.3, total_time*0.4],
                                    burst_factor=4,)
    posC = trackC['XYZ']
    
    
    # --- Simple visualization ---
    plt.close('all')
    fig = plt.figure(figsize=(12,4))    

    ax1 = fig.add_subplot(131) #, projection='3d')
    ax1.plot(posA[:,0], posA[:,1], marker='.', label="Scenario A")
    ax1.set_title("A: Basic Random + Reflect\n(no layering, no events)")

    ax2 = fig.add_subplot(132) #, projection='3d')
    ax2.plot(posB[:,0], posB[:,1], marker='.', color='red', label="Scenario B")
    ax2.set_title("B: Layered + Turn+Accel Events")

    ax3 = fig.add_subplot(133,)
    ax3.plot(posC[:,0], posC[:,1], marker='.', color='green', label="Scenario C")
    ax3.set_title("C: 'burst' Mode + Turn Events")

    plt.tight_layout()
    plt.show()


