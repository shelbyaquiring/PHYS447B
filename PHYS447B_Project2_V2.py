#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as sci
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.animation as animation
import time


# Class definition for storing an orbit solution
class Orbit:
    def __init__(self, x, y, theta, t, solver=None, error=None, soltime=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.theta = np.array(theta)
        self.t = np.array(t)
        self.solver = solver
        self.error = error
        self.soltime = soltime       

        # Calculate derivatives
        self.tdot = self.t[:-1] # Time vector minus the final element
        self.xdot = np.diff(self.x)/np.diff(self.t)
        self.ydot = np.diff(self.y)/np.diff(self.t)
        self.vmag = np.sqrt(self.xdot**2 + self.ydot**2)
        

def plot_3D_orbits(rvec, t, R, live=False):
    # Figure setup
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', computed_zorder=False)

    # Draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = R * np.cos(u) * np.sin(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(v)
    ax.plot_surface(x, y, z, cmap="Blues", alpha=0.5, edgecolors='tab:blue', zorder=0)

    # Plot orbital path
    pathlist = []
    positionlist = []
    if np.shape(rvec)[0] > 1:
        for idx, r in enumerate(rvec):
            path, = ax.plot3D(r[0][0], r[1][0], r[2][0], label=f'Trajectory {idx}', alpha=0.8, zorder=0)
            position, = ax.plot3D(r[0][0], r[1][0], r[2][0], '.', markersize=10, label='Initial Position', zorder=1)
            pathlist.append(path)
            positionlist.append(position)
    else:
        r = rvec[0]
        path, = ax.plot3D(r[0][0], r[1][0], r[2][0], label=f'Trajectory', alpha=0.8, zorder=0)
        position, = ax.plot3D(r[0][0], r[1][0], r[2][0], '.', markersize=10, label='Initial Position', zorder=1)
        pathlist.append(path)
        positionlist.append(position)

    # Add axis vectors
    l = 2*R
    start = [0,0,0]
    x,y,z = [start,start,start]
    u,v,w = [[l,0,0],[0,l,0],[0,0,l]]
    ax.quiver(x,y,z,u,v,w, color='k', alpha=0.5, arrow_length_ratio=0.1)
    
    # Add labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    minval = 0
    maxval = 0
    for r in rvec:
        maxval = np.max([maxval, np.max(r[0]), np.max(r[1]), np.max(r[2])])
        minval = np.min([minval, np.min(r[0]), np.min(r[1]), np.min(r[2])])
    ax.set_xlim([minval, maxval])
    ax.set_ylim([minval, maxval])
    ax.set_zlim([minval, maxval])

    ax.legend()

    # If plot is commanded to live update 
    if live:

        # Plot orbiting body
        frameSpeed = 20 # ms btw frames
        speed = 5000 # Multiplier

        # Defining interpolation functions for each axis
        fxlist = []
        fylist = []
        fzlist = []
        for idx, r in enumerate(rvec):
            fx = interp1d(t[idx], r[0], kind='cubic', fill_value='extrapolate')
            fy = interp1d(t[idx], r[1], kind='cubic', fill_value='extrapolate')
            fz = interp1d(t[idx], r[2], kind='cubic', fill_value='extrapolate')
            fxlist.append(fx)
            fylist.append(fy)
            fzlist.append(fz)

        # Defining initializing function because the animator requires 
        # this syntax and breaks without it in 3D
        def init():
            return *positionlist, *pathlist,

        # Defining animation generator function
        t0 = time.time()
        xvec = []
        yvec = []
        zvec = []
        for idx, _ in enumerate(rvec): # Initialize trajectory vectors
            xvec.append([])
            yvec.append([])
            zvec.append([])

        def animate(i):            
            for idx, _ in enumerate(rvec):
                T = (speed * i * frameSpeed / 1000) % t[idx][-1]            
                x_c = fxlist[idx](T)
                y_c = fylist[idx](T)
                z_c = fzlist[idx](T)
                positionlist[idx].set_data_3d(x_c, y_c, z_c) # update the data.
                xvec[idx].append(np.copy(x_c))
                yvec[idx].append(np.copy(y_c))
                zvec[idx].append(np.copy(z_c))  
                pathlist[idx].set_data_3d(xvec[idx], yvec[idx], zvec[idx])                       
            return *positionlist, *pathlist,

        # Animating plot
        ani = animation.FuncAnimation(
            fig, animate, init_func=init, interval=frameSpeed, repeat=True, blit=True, save_count=50)

    plt.show()

    return fig, ax, ani

def plot_3D_orbit(rvec, t, R, live=False):
    # Figure setup
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', computed_zorder=False)

    # Draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = R * np.cos(u) * np.sin(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(v)
    ax.plot_surface(x, y, z, cmap="Blues", alpha=0.5, edgecolors='tab:blue', zorder=0)

    # Plot orbital path
    path, = ax.plot3D(rvec[0][0], rvec[1][0], rvec[2][0], label='Trajectory', alpha=0.8, zorder=0)
    position, = ax.plot3D(rvec[0][0], rvec[1][0], rvec[2][0], '.', markersize=10, label='Initial Position', zorder=1)

    # Add axis vectors
    l = 2*R
    start = [0,0,0]
    x,y,z = [start,start,start]
    u,v,w = [[l,0,0],[0,l,0],[0,0,l]]
    ax.quiver(x,y,z,u,v,w, color='k', alpha=0.5, arrow_length_ratio=0.1)
    
    # Add labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    maxval = np.max([np.max(rvec[0]), np.max(rvec[1]), np.max(rvec[2])])
    minval = np.min([np.min(rvec[0]), np.min(rvec[1]), np.min(rvec[2])])
    ax.set_xlim([minval, maxval])
    ax.set_ylim([minval, maxval])
    ax.set_zlim([minval, maxval])

    ax.legend()

    # If plot is commanded to live update 
    if live:

        # Plot orbiting body
        frameSpeed = 20 # ms btw frames
        speed = 5000 # Multiplier

        # Defining interpolation functions for each axis
        fx = interp1d(t, rvec[0], kind='cubic', fill_value='extrapolate')
        fy = interp1d(t, rvec[1], kind='cubic', fill_value='extrapolate')
        fz = interp1d(t, rvec[2], kind='cubic', fill_value='extrapolate')

        # Defining initializing function because the animator requires 
        # this syntax and breaks without it in 3D
        def init():
            return position, path,

        # Defining animation generator function
        t0 = time.time()
        xvec = []
        yvec = []
        zvec = []
        def animate(i):
            T = (speed * i * frameSpeed / 1000) % t[-1]            
            x_c = fx(T)
            y_c = fy(T)
            z_c = fz(T) 
            position.set_data_3d(x_c, y_c, z_c) # update the data.
            xvec.append(x_c)
            yvec.append(y_c)
            zvec.append(z_c)  
            path.set_data_3d(xvec, yvec, zvec)         
            return position, path,

        # Animating plot
        ani = animation.FuncAnimation(
            fig, animate, init_func=init, interval=frameSpeed, repeat=True, blit=True, save_count=50)

    plt.show()

    return fig, ax, ani





# Define the exact solution aproach
def exactSolution(e, rp, mu, n):
    # Define true anomaly vector
    theta = np.linspace(0, 2*np.pi, n)

    # Calculate angular momntum
    h = np.sqrt((e+1) * rp * mu)

    # Calculate orbital positions
    r = h**2/mu * 1/(1 + e*np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Calculate time
    time = [0] + list(h**3 / mu**2 * sci.cumtrapz(1/(1 + e*np.cos(theta))**2, theta))

    return x, y, theta, time  


# Define the numerical solution approach
# def numericalSolution(state_i, e, rp, mu, reltol, T, solver, oblate=False):
  
# https://en.wikipedia.org/wiki/Geopotential_model#:~:text=J2%20%3D%201.75553%20%C3%97%201010%20km5%E2%8B%85s&text=J3%20%3D%20%E2%88%922.61913%20%C3%97%201011%20km6%E2%8B%85s&text=For%20example%2C%20at%20a%20radius,the%20order%20of%202%20permille.
def geopotentialFunc(r, t, q, R):
    """
    R: Radius from Earth centre to satellite
    r: Radiu fro earth centre to some point within Earth at which density is being evaluated
    t: Theta at which density is evaluated
    q: Phi at which density is evaluated
    """
    # Calculate density
    rho_0 = 4000 # kg/m**3
    drho = 100
    rho = rho_0 + drho * np.sin(t) # Density at this location

    # Calculate difference between satellite position and position within earth
    x = r * np.sin(t) * np.cos(q)
    y = r * np.sin(t) * np.sin(q)
    z = r * np.cos(t)
    d = R - np.array([x, y, z])
    dhat = d / np.linalg.norm(d)

    # Calculate integrand for geopotential 
    A = rho / np.linalg.norm(d)**2 * dhat
    return A



def numericalSolution(initial_state, mu, reltol, T, solver, oblate=False):
    # # Calculate angular momntum
    # h = np.sqrt((e+1) * rp * mu)

    # # Calculate initial velocity
    # v0 = h / rp
    
    # # Numerical parameters
    # initial_state = [rp, 0, 0, 0, v0/1.4, v0/1.4]

    def state_derivative(time, state):
        # Extract position and velocity vectors
        r = state[:DIMENSIONS] # Position
        v = state[DIMENSIONS:] # Velocity
        
        if oblate: # If oblateness is being considered
            # rvec = np.linspace(0, Re, N)
            # thetavec = np.linspace(0, np.pi, N)
            # phivec = np.linspace(0, 2*np.pi, N)

            # dr = rvec[1] - rvec[0]
            # dt = thetavec[1] - thetavec[0]
            # dq = phivec[1] - phivec[0]

            # a = np.zeros(3) # acceleration, m/s**2
            # for ri in rvec:
            #     for ti in thetavec:
            #         for qi in phivec:
            #             # Calculate cartesian coords of integration point

                        
            #             # Calculate distance between orbiting body and integration point
            #             dx = r[0][0] - xi
            #             dy = r[1][0] - yi
            #             dz = r[2][0] - zi
            #             d = np.sqrt(dx**2 + dy**2 + dz**2)
            #             dhat = np.array([dx, dy, dz]) / d

            #             # Calculate incremental acceleration
            #             rho = rho_0 + drho * np.sin(ti) # Density at this location
            #             da = rho / d**2 * ri**2 * np.sin(ti) * dr * dt * dq
            #             a = a + da * dhat

            A = sci.tplquad(geopotentialFunc, 0, Re, 0, np.pi, 0, 2*np.pi, args=(r,))

            # Account for -G
            print(A)
            A *= -G   
            a = list(A) # Format for unpacking in function return       


        else:
            # Calculate the ideal 2-body component
            a = - (mu / np.linalg.norm(r)**3) * r

        

        return [*v, *a]

    sol = sci.solve_ivp(state_derivative, [0, T], initial_state, method=solver, vectorized=True, rtol=reltol)
    theta = np.arctan2(sol.y[1], sol.y[0]) # Calculate true anomaly

    # Correct negative true anomaly
    for idx, angle in enumerate(theta):
        if angle < 0:
            theta[idx] += np.pi*2

    return sol.y[0], sol.y[1], sol.y[2], theta, sol.t


class Orbit:
    def __init__(self, x, y, z, theta, t, solver=None, error=None, soltime=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.theta = np.array(theta)
        self.t = np.array(t)
        self.solver = solver
        self.error = error
        self.soltime = soltime       

        # Calculate derivatives
        self.tdot = self.t[:-1] # Time vector minus the final element
        self.xdot = np.diff(self.x)/np.diff(self.t)
        self.ydot = np.diff(self.y)/np.diff(self.t)
        self.zdot = np.diff(self.z)/np.diff(self.t)
        self.vmag = np.sqrt(self.xdot**2 + self.ydot**2 + self.zdot**2)



if __name__ == "__main__":
    # Inputs
    Re = 6371e3 # m, Earth radius
    Me = 5.97e24 # kg, Earth mass
    G = 6.67e-11 # Gravitational constant

    # # Inputs
    DIMENSIONS = 3 # 2D sim right now
    perigee = 1000e3 # m
    # evec = [0, 0.3, 0.7, 1] # e = h**2/mu * 1/rp -1
    # solver_list = ['RK45',  'LSODA'] #  'Radau', 'BDF',
    # N = 1000 # Number of points over which to evaluate the exact solution
    # T = 5e4 # Time period over which to evaluate the numerical solution
    # reltol = 2.220446049250313e-14 # Relative tolerance for numerical solution

    solver = 'LSODA'
    reltol = 1e-3
    e = 0.5
    T = 1e5 # Time period over which to evaluate the numerical solution

    # Intermediates
    rp = Re + perigee
    mu = G * Me
    
    r0 = rp # km
    v0 = 8000

    solvec = []
    timevec = []
    n = 2
    for _ in range(n):
        rvec = np.array([1, 0, 0]) + np.random.rand(3) * 0.01
        rvec_0 = r0 * (rvec / np.linalg.norm(rvec))
        vvec = np.array([0, 1, 1]) + np.random.rand(3) * 0.01
        vvec_0 = v0 * (vvec / np.linalg.norm(vvec))
        s0 = [*rvec_0, *vvec_0]
        # Calculating numerical solution
        x_n, y_n, z_n, theta_n, time_n = numericalSolution(s0, mu, reltol, T, solver, oblate=True)
        sol_n = Orbit(x_n, y_n, z_n, theta_n, time_n, solver)
        solvec.append([x_n, y_n, z_n])
        timevec.append(time_n)

    # x_n2, y_n2, z_n2, theta_n2, time_n2 = numericalSolution(e, rp*2, mu, reltol, T, solver)
    # sol_n2 = Orbit(x_n, y_n, z_n, theta_n, time_n, solver)

    # Plotting
    # thetavec = np.linspace(0,np.pi*2,1000)
    # x = 1.1*Re * np.cos(thetavec)
    # y = 1.1*Re * np.sin(thetavec)
    # z = 1.1*Re * np.zeros_like(x)
    # position = [[x_n,y_n,z_n], [x_n2,y_n2,z_n2]]
    # position2 = [x_n2,y_n2,z_n2]
    plot_3D_orbits(solvec, timevec, Re, live=True)
    #plot_3D_orbit(position2, time_n2, Re, live=True)

    

    #plt.show()


