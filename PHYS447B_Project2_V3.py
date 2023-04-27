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

#https://conference.sdo.esoc.esa.int/proceedings/sdc6/paper/130/SDC6-paper130.pdf
#https://arxiv.org/pdf/1309.5244.pdf
#https://www.spaceacademy.net.au/watch/debris/atmosmod.htm

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

def plot_3D_orbits(orbits, R, live=False):
    # Figure setup
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', computed_zorder=False)

    # Draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xx = R * np.cos(u) * np.sin(v)
    yy = R * np.sin(u) * np.sin(v)
    zz = R * np.cos(v)
    ax.plot_surface(xx, yy, zz, cmap="Blues", alpha=0.5, edgecolors='tab:blue', zorder=0)

    # Plot orbital path
    pathlist = []
    positionlist = []
    if np.shape(orbits)[0] > 1: # If thre's more than 1 orbit to plot
        for idx, orb in enumerate(orbits):
            path, = ax.plot3D(orb.x[0], orb.y[0], orb.z[0], label=f'Trajectory {idx}', alpha=0.8, zorder=0)
            position, = ax.plot3D(orb.x[0], orb.y[0], orb.z[0], '.', markersize=10, label='Initial Position', zorder=1)
            pathlist.append(path)
            positionlist.append(position)
    else:
        orb = orbits[0]
        path, = ax.plot3D(orb.x[0], orb.y[0], orb.z[0], label=f'Trajectory', alpha=0.8, zorder=0)
        position, = ax.plot3D(orb.x[0], orb.y[0], orb.z[0], '.', markersize=10, label='Initial Position', zorder=1)
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
    for orb in orbits:
        maxval = np.max([maxval, np.max(orb.x), np.max(orb.y), np.max(orb.z)])
        minval = np.min([minval, np.min(orb.x), np.min(orb.y), np.min(orb.z)])
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
        for idx, orb in enumerate(orbits):
            fx = interp1d(orb.t, orb.x, kind='cubic', fill_value='extrapolate')
            fy = interp1d(orb.t, orb.y, kind='cubic', fill_value='extrapolate')
            fz = interp1d(orb.t, orb.z, kind='cubic', fill_value='extrapolate')
            fxlist.append(fx)
            fylist.append(fy)
            fzlist.append(fz)

        # Defining initializing function because the animator requires 
        # this syntax and breaks without it in 3D
        def init():
            return *positionlist, *pathlist,

        # Defining animation generator function
        xvec = []
        yvec = []
        zvec = []
        for idx, _ in enumerate(orbits): # Initialize trajectory vectors
            xvec.append([])
            yvec.append([])
            zvec.append([])

        def animate(i):            
            for idx, orb in enumerate(orbits):
                T = (speed * i * frameSpeed / 1000) % orb.t[-1]            
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

def plot_altitude(orbits, oblate=None, drag=None, perigee=None):
    # Figure setup
    fig, ax = plt.subplots()

    # Plot orbital path
    for idx, orb in enumerate(orbits):
        labelstr = f'Trajectory {idx}'
        if oblate != None:
            if oblate[idx]:
                labelstr += ', Oblate'
            else:
                labelstr += ', Not Oblate'
        if drag != None:
            if drag[idx]:
                labelstr += ', Drag'
            else:
                labelstr += ', No Drag'

        # labelstr += f', e={e[idx]}'
        
        altitude = (np.sqrt(np.array(orb.x)**2 + np.array(orb.y)**2 + np.array(orb.z)**2) - Re) * 1e-3
        path = ax.plot(orb.t, altitude, label=labelstr, alpha=0.8, zorder=0)

    # Add labels
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title('Altitude vs Time')
    ax.grid('enable')
    ax.legend()

    return fig, ax


def plot_orbital_elements(orbits, oblate=False, drag=False):
    # Now, plot
    fig, ax = plt.subplots(3, 2)

    # Plot orbital path
    for idx, orb in enumerate(orbits):
        labelstr = f'Trajectory {idx}'
        if oblate != None:
            if oblate[idx]:
                labelstr += ', Oblate'
            else:
                labelstr += ', Not Oblate'
        if drag != None:
            if drag[idx]:
                labelstr += ', Drag'
            else:
                labelstr += ', No Drag'

        # labelstr += f', e={e[idx]}'

        ax[0][0].plot(orb.t, orb.h, label=labelstr)#f'{idx}')
        ax[0][1].plot(orb.t, orb.e)
        ax[1][0].plot(orb.t, orb.omega)
        ax[1][1].plot(orb.t, orb.i)
        ax[2][0].plot(orb.t, orb.w)
        ax[2][1].plot(orb.t, orb.theta)


    # Add labels
    xlabel = 'Time [s]'
    ylabels = ['[m**2/s]', '[-]', '[rad]', '[rad]', '[rad]', '[rad]']
    titles = ['Angular Momentum', 'Eccentricity', 'RAAN', 'Inclination', 'Arg. Perigee', 'True Anomaly']
    for idx in range(6):
        row = idx // 2
        col = idx % 2
        ax[row][col].set_xlabel(xlabel)
        ax[row][col].set_ylabel(ylabels[idx])
        ax[row][col].set_title(titles[idx])
        ax[row][col].grid('enable')
    
    ax[0][0].legend()

    fig.tight_layout()

    return fig, ax


def plot_perifocal(orbits, oblate=False, drag=False, altitude=False):
    # Define figure
    fig, ax = plt.subplots()

    # Plot orbital path in perifocal frame
    for idx, orb in enumerate(orbits):
        labelstr = f'Trajectory {idx}'
        if oblate != None:
            if oblate[idx]:
                labelstr += ', Oblate'
            else:
                labelstr += ', Not Oblate'
        if drag != None:
            if drag[idx]:
                labelstr += ', Drag'
            else:
                labelstr += ', No Drag'

        # labelstr += f', e={e[idx]}'

        if altitude:
            ax.plot(orb.xp - Re*np.cos(orb.theta), orb.yp - Re*np.sin(orb.theta), alpha=0.5, label=labelstr)
        else:
            ax.plot(orb.xp, orb.yp, alpha=0.5, label=labelstr)

    # Add labels
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    ax.set_title('Orbital Comparison: Perifocal Frame')
    ax.grid('enable')
    
    ax.legend()
    ax.set_aspect('equal')
    fig.tight_layout()

    return fig, ax


def numericalSolution(initial_state, mu, reltol, T, solver, oblate=False, drag=False):

    # Define the state derivative function
    def state_derivative(time, state):
        # Extract position and velocity vectors
        r = state[:DIMENSIONS] # Position

        if not np.linalg.norm(r) > Re: # If orbit is below the surface of the earth, no need to solve
            v = [0, 0, 0]
            a = [0, 0, 0]

        else:
            v = state[DIMENSIONS:] # Velocity            
            a = - (mu / np.linalg.norm(r)**3) * r # Calculate the ideal 2-body monopole term
            
            if oblate: # If oblateness is being considered
                # https://en.wikipedia.org/wiki/Geopotential_model
                J2 = 1.0826e-3
                K = J2  * mu * Re**2 # Lumped constants term
                rmag = np.linalg.norm(r)
                ax = K * r[0] / rmag**7 * (6*r[2]**2 - 3/2*(r[0]**2 + r[1]**2))
                ay = K * r[1] / rmag**7 * (6*r[2]**2 - 3/2*(r[0]**2 + r[1]**2))
                az = K * r[2] / rmag**7 * (3*r[2]**2 - 9/2*(r[0]**2 + r[1]**2))
                da = np.array([ax, ay, az])
                #print(da / a)
                a = a + da

            if drag: # if drag is being considered
                Cd = 2.2
                A = 1 # m**2
                H = 7000 #1.386e-4 # Height scale factor, m
                m = 1 # kg, satellite mass
                h = np.linalg.norm(r) - Re # Height off the earth's surface, m
                if h < 0:
                    da = 0
                else:
                    rho = 1.225 * np.exp(-h/H) # kg/m**3
                    F = 1/2 * rho * Cd * A * np.linalg.norm(v)**2
                    vhat = np.array(v) / np.linalg.norm(v)
                    da = - (F / m) * vhat

                a = a + da
           
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

if __name__ == "__main__":
    # Inputs
    Re = 6371e3 # m, Earth radius
    Me = 5.97e24 # kg, Earth mass
    G = 6.67e-11 # Gravitational constant

    # # Inputs
    DIMENSIONS = 3 # 2D sim right now
    perigee = 200e3 # m

    solver = 'LSODA'
    reltol = 1e-14
    e = 0.0001
    #e = np.array([0.01, 0.1, 0.3, 0.5])
    T = 2e4 # Time period over which to evaluate the numerical solution

    # Intermediates
    rp = Re + perigee
    mu = G * Me
    
    r0 = rp # m

    # Calculate angular momntum
    h = np.sqrt((e+1) * rp * mu)
    
    # Calculate initial velocity
    v0 = h / rp
    #v0 = np.sqrt(mu/r0) # circular

    # Create lists to hold solutions
    solvec = []
    timevec = []
    i = 98 # SSO inclination
    
    obvec = [False, False, True, True]
    drag = [False, True, False, True]
    # obvec = [False]*4
    # drag = [False]*4
    for idx in range(len(obvec)):
        rvec = np.array([1, 0, 0])                              
        rvec_0 = r0 * (rvec / np.linalg.norm(rvec))
        vvec = np.array([0, np.cos(i*np.pi/180), np.sin(i*np.pi/180)]) 
        vvec_0 = v0 * (vvec / np.linalg.norm(vvec))
        s0 = [*rvec_0, *vvec_0]
        # Calculating numerical solution
        # x_n, y_n, z_n, theta_n, time_n = numericalSolution(s0, mu, reltol, T, solver, oblate=obvec[idx], drag=drag[idx])
        orbit_n = numericalSolution(s0, mu, reltol, T, solver, oblate=obvec[idx], drag=drag[idx])   
        solvec.append(orbit_n)


    # Call plotting functions
    plot_3D_orbits(solvec, Re, live=True)
    plot_altitude(solvec, oblate=obvec, drag=drag, perigee=perigee)
    plot_orbital_elements(solvec, oblate=obvec, drag=drag)
    plot_perifocal(solvec, oblate=obvec, drag=drag)

    plt.show()
    

