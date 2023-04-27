#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as sci
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.animation as animation
from scipy.optimize import root
from scipy.integrate import cumtrapz
import math


# Class definition for storing an orbit solution
class Orbit3D:
    """
    Input x, y, z coordinates in geocentric equatorial frame. 
    Cartesian velocity and accelerations are optional inputs.
    Classical orbital elements are also optional inputs.
    """
    def __init__(self, x, y, z, t, vx=None, vy=None, vz=None, ax=None, ay=None, az=None):
        # Store cartesian position in geocentric equatorial frame
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        # Store cartesian velocity in geocentric equatorial frame
        self.vx = np.array(vx)
        self.vy = np.array(vy)
        self.vz = np.array(vz)
        # Store cartesian acceleration in geocentric equatorial frame
        self.ax = np.array(ax)
        self.ay = np.array(ay)
        self.az = np.array(az)
        # Store time
        self.t = np.array(t)
        # Calculate classical orbital elements if velocity is provided
        if vx is not None: # If we have all velocity vectors
            # Initialize arrays for orbital elements
            h = np.zeros(len(x))
            e = np.zeros(len(x))
            omega = np.zeros(len(x))
            i = np.zeros(len(x))
            w = np.zeros(len(x))
            theta = np.zeros(len(x))
            xp = np.zeros(len(x))
            yp = np.zeros(len(x))
            vxp = np.zeros(len(x))
            vyp = np.zeros(len(x))

            for idx in range(len(x)):
                # Calculate orbital elements for this timestep
                rvec = [x[idx], y[idx], z[idx]]
                vvec = [vx[idx], vy[idx], vz[idx]]
                h_i, e_i, omega_i, i_i, w_i, theta_i = coe_from_sv(rvec, vvec, mu)
                # Assign orbital element instance to vectors
                h[idx] = h_i
                e[idx] = e_i
                omega[idx] = omega_i
                i[idx] = i_i
                w[idx] = w_i
                theta[idx] = theta_i
                # Calculate perifocal frame vectors
                _, _, rp, vp = sv_from_coe(h_i, e_i, omega_i, i_i, w_i, theta_i, mu)
                xp[idx] = rp[0]
                yp[idx] = rp[1]
                vxp[idx] = vp[0]
                vyp[idx] = vp[1]

            # Assign orbtal element vectors to the objects
            self.h = h
            self.e = e
            self.omega = np.unwrap(omega)
            self.i = np.unwrap(i)
            self.w = np.unwrap(w)
            self.theta = np.unwrap(theta)
            self.xp = xp
            self.yp = yp
            self.vxp = vxp
            self.vyp = vyp


            
        
        else: # If we don't have the velocity, generate the fields but assign None
            self.h = None
            self.e = None
            self.omega = None
            self.i = None
            self.w = None
            self.theta = None
            self.rp = None
            self.vp = None



def sv_from_coe(h, e, omega, i, w, theta, mu):
    # Calculate the perifocal position and velocity
    rp = h**2/mu * 1/(1 + e*np.cos(theta)) * np.array([np.cos(theta), np.sin(theta), 0])
    vp = mu/h * np.array([-np.sin(theta), e + np.cos(theta), 0])

    # Calculate rotation matricies
    R_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])

    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]        
    ])

    R_w = np.array([
        [np.cos(w), -np.sin(w), 0],
        [np.sin(w), np.cos(w), 0],
        [0, 0, 1]
    ])

    # Combine rotation matricied
    Q_1 = np.matmul(R_omega, R_i)
    Q_px = np.matmul(Q_1, R_w)

    # Calculate position and velocity in geocentric equatorial frame
    rg = np.matmul(Q_px, rp)
    vg = np.matmul(Q_px, vp)

    return rg, vg, rp, vp




def coe_from_sv(rvec, vvec, mu):
    """
    INPUTS: 
    - Geocentric equatorial postion vector (x, y, z)
    - Geocentric equatorial velocity vector (vx, vy, vz) 
    - Gravitational parameter mu

    OUTPUTS:
    - h: Specific Angular Momentum [m**2/s]
    - e: Eccentricity [unitless] 
    - omega: Right Ascension of the Ascending Node [rad]
    - i: Inclination [rad]
    - w: Argument of Perigee [rad]
    - theta: True Anomaly [rad]
    """
    rvec = np.array(rvec)
    vvec = np.array(vvec)
    # Caculate magnitude of position and velocity
    r = np.linalg.norm(rvec)
    v = np.linalg.norm(vvec)

    # Calculate radial velocity
    vr = np.dot(rvec, vvec) / r

    # Calculate angular momentum 
    hvec = np.cross(rvec, vvec)
    h = np.linalg.norm(hvec)

    # Calculate inclination
    i = np.arccos(hvec[-1]/h)

    # Calculate node line vector
    nvec = np.cross([0, 0, 1], hvec)
    n = np.linalg.norm(nvec)

    # Calculate right ascenscion of ascending node
    if n != 0:
        omega = np.arccos(nvec[0]/n)
        if nvec[1] < 0:
            omega = 2*np.pi - omega
    else:
        omega = 0
        
    # Calculate eccentricity
    evec = 1/mu * ((v**2 - mu/r) * rvec - r*vr*vvec)
    e = np.linalg.norm(evec)

    # Calculate argument of perigee
    if n != 0:
        w = np.arccos(np.dot(nvec, evec)/(n*e))
        if evec[-1] < 0:
            w = 2*np.pi - w
    else:
        w = 0

    # Calculate true anomaly
    theta = np.arccos(np.dot(evec, rvec)/(e*r))
    # THE CORRECTION BELOW IS REMOVED BECAUSE PROBLEMS 
    # HAPPEN WHEN OBLATENESS IS ADDED
    if vr < 0:
        theta = 2*np.pi - theta


    return h, e, omega, i, w, theta



def numericalSolution(initial_state, mu, reltol, T, solver):
    # Define the state derivative function
    def state_derivative(time, state):
        # Extract position and velocity vectors
        r = state[:DIMENSIONS] # Position

        if not np.linalg.norm(r) > Re: # If orbit is below the surface of the earth, no need to solve
            v = [0, 0, 0]
            a = [0, 0, 0]

        else:
            v = state[DIMENSIONS:] # Velocity            
            a = - (mu / np.linalg.norm(r)**3) * r  + thrust(time, r, v) / Msc # Acceleration from gravity and propulsion system
            # print(f'IVP mean a: {np.mean(a)}')
        return [*v, *a]
    
    # Calculate numerical solution with solve_ivp
    sol = sci.solve_ivp(state_derivative, [0, T], initial_state, method=solver, vectorized=True, rtol=reltol)
    theta = np.arctan2(sol.y[1], sol.y[0]) # Calculate true anomaly

    # Correct negative true anomaly
    for idx, angle in enumerate(theta):
        if angle < 0:
            theta[idx] += np.pi*2

    # Store as list of orbits
    out = Orbit3D(x=sol.y[0], y=sol.y[1], z=sol.y[2], vx=sol.y[3], vy=sol.y[4], vz=sol.y[5], t=sol.t)

    # return sol.y[0], sol.y[1], sol.y[2], theta, sol.t
    return out




def numericalSolutionBVP(initial_state, final_state, mu, reltol, T, sol_ivp = None):
    
    # Define the state derivative function
    def state_derivative(time, state, params):
        # Extract position and velocity vectors
        r = state[:3] # Position

        # if not np.linalg.norm(r) > Re: # If orbit is below the surface of the earth, no need to solve
        #     v = [0, 0, 0]
        #     a = [0, 0, 0]
        thrust1 = params[:3]
        thrust2 = params[3:]

        # else:
        v = state[3:] # Velocity
        radius = np.array([np.linalg.norm(x) for x in np.transpose(r)])       
        a = - (mu / radius**3) * r  #+  / Msc # Acceleration from gravity and propulsion system
        thrust_input = np.zeros_like(a)
        thrust_input[:,0] = thrust1 #thrust1 * v[:,0] / np.linalg.norm(v[:,0])
        thrust_input[:,-1] = thrust2 #thrust2 * v[:,-1] / np.linalg.norm(v[:,-1])

        #np.transpose(np.squeeze([thrust(time, np.transpose(r)[idx], np.transpose(v)[idx]) for idx in range(len(r[0]))]))
        # print(np.shape(delta))
        a += thrust_input
        out = np.vstack((*v, *a))
        return out
    
    # def bc(ya, yb):
    #     initial = ya[:3] - initial_state[:3]
    #     final = yb[:3] - final_state[:3]
    #     return np.array((*initial, *final))

    def bc(ya, yb, params):
        initial = ya - initial_state
        final = yb - final_state
        return np.array((*initial, *final))

    # Calculate numerical solution with solve_ivp
    timevec = np.linspace(0, T, 3)
    yout = np.ones((6, timevec.size))
    yout[:,0] = initial_state
    yout[:,-1] = final_state
    # yout[2,:] = [0, 0, 0]s
    sol = sci.solve_bvp(state_derivative, bc, timevec, yout, p=[1, -1, 0, 1, -1, 0], tol=1e-5, max_nodes=20000)

    print(sol)

    # Store as list of orbits
    out = Orbit3D(x=sol.y[0], y=sol.y[1], z=sol.y[2], vx=sol.y[3], vy=sol.y[4], vz=sol.y[5], t=sol.x)

    return out


def thrust(time, r, v):
    """
    Function calculating the thrust vector from propulsion system.
    """
    # If statements below define the condition for firing
    vy_hat = np.dot(np.squeeze(v), [0, 1, 0])/np.linalg.norm(v) # projection of velocity onto y vector
    if np.abs(vy_hat - 1) < 1e-2:
        # return Thrust * np.array([[0], [1], [0]])
        return np.array([[0], [0], [0]])
    else:
        return np.array([[0], [0], [0]])


if __name__ == "__main__":
    # Inputs
    Re = 6371e3 # m, Earth radius
    Me = 5.97e24 # kg, Earth mass
    G = 6.67e-11 # Gravitational constant

    Msc = 1000 # kg, spacecraft mass
    Thrust = 500 # N, engine thrust

    # Inputs
    DIMENSIONS = 3 # 2D sim right now

    perigee = 400e3 # m
    i = 10 * np.pi / 180 # Inclin.
    w = 50 * np.pi / 180 # AoP
    omega = 10 * np.pi / 180 # RAAN
    solver = 'LSODA'
    reltol = 1e-14
    e = 0.1
    T = 5e3 # Time period over which to evaluate the numerical solution

    # Intermediates
    rp = Re + perigee
    mu = G * Me
    
    r0 = rp # m
    v0 = 8000

    rvec = np.array([1, 0, 0])                              
    rvec_0 = r0 * (rvec / np.linalg.norm(rvec))
    vvec = np.array([0, 1, 0])#np.array([0, np.cos(i*np.pi/180), np.sin(i*np.pi/180)]) 
    vvec_0 = v0 * (vvec / np.linalg.norm(vvec))
    s0 = [*rvec_0, *vvec_0]
    s02 = [*rvec_0, *vvec_0*1.185]

    # Calculating numerical solution
    # x_n, y_n, z_n, theta_n, time_n = numericalSolution(s0, mu, reltol, T, solver, oblate=obvec[idx], drag=drag[idx])
    orbit_n = numericalSolution(s0, mu, reltol, T, solver)
    
    orbit_n2 = numericalSolution(s02, mu, reltol, T, solver)
    s1 = [orbit_n2.x[-1], orbit_n2.y[-1], orbit_n2.z[-1], orbit_n2.vx[-1], orbit_n2.vy[-1], orbit_n2.vz[-1]]
    orbit_n_bvp = numericalSolutionBVP(s0, s1, mu, reltol, T)   

    # Plot output
    fig, ax = plt.subplots()
    ax.plot(orbit_n.x, orbit_n.y, alpha=0.5, label='IVP a')
    ax.plot(orbit_n2.x, orbit_n2.y, alpha=0.5, label='IVP b')

    ax.plot(orbit_n_bvp.x, orbit_n_bvp.y, '--', label='BVP')

    ax.set_aspect('equal')
    ax.grid('enable')
    ax.legend()

    # Plot output
    fig, ax = plt.subplots()
    ax.plot(orbit_n.t, orbit_n.x, alpha=0.5, label='IVP X')
    ax.plot(orbit_n.t, orbit_n.y, alpha=0.5, label='IVP Y')
    

    ax.plot(orbit_n_bvp.t, orbit_n_bvp.x, '--', label='BVP X')
    ax.plot(orbit_n_bvp.t, orbit_n_bvp.y, '--', label='BVP Y')
    
    

    fig, ax = plt.subplots()
    ax.plot(orbit_n.t, orbit_n.vx, alpha=0.5, label='IVP Vx')
    ax.plot(orbit_n.t, orbit_n.vy, alpha=0.5, label='IVP Vy')

    ax.plot(orbit_n_bvp.t, orbit_n_bvp.vx, '--', label='BVP Vx')
    ax.plot(orbit_n_bvp.t, orbit_n_bvp.vy, '--', label='BVP Vy')

    ax.grid('enable')   
    ax.legend()

# %%
