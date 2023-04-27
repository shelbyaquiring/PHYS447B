#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import scipy.integrate as sci
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.animation as animation
import time


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
def numericalSolution(e, rp, mu, reltol, T, solver):
    # Calculate angular momntum
    h = np.sqrt((e+1) * rp * mu)

    # Calculate initial velocity
    v0 = h / rp
    
    # Numerical parameters
    initial_state = [rp, 0, 0, 0, v0/1.4, v0/1.4]

    def state_derivative(time, state):
        r = state[:DIMENSIONS] # Position
        v = state[DIMENSIONS:] # Velocity
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
    reltol = 1e-6
    e = 0.5
    T = 1e5 # Time period over which to evaluate the numerical solution

    # Intermediates
    rp = Re + perigee
    mu = G * Me
    
    # Calculating numerical solution
    x_n, y_n, z_n, theta_n, time_n = numericalSolution(e, rp, mu, reltol, T, solver)
    sol_n = Orbit(x_n, y_n, z_n, theta_n, time_n, solver)

    # Plotting
    # thetavec = np.linspace(0,np.pi*2,1000)
    # x = 1.1*Re * np.cos(thetavec)
    # y = 1.1*Re * np.sin(thetavec)
    # z = 1.1*Re * np.zeros_like(x)
    position = [x_n,y_n,z_n]
    plot_3D_orbit(position, time_n, Re, live=True)

    

    #plt.show()


