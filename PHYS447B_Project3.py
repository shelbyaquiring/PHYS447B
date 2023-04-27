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

    # If plot is commanded to live update 
    if not live:
        ani = None

        # Plot full orbital path
        pathlist = []
        positionlist = []
        if np.shape(orbits)[0] > 1: # If thre's more than 1 orbit to plot
            for idx, orb in enumerate(orbits):
                path = ax.plot3D(orb.x, orb.y, orb.z, label=f'Trajectory {idx}', alpha=0.8, zorder=0)
                position = ax.plot3D(orb.x[0], orb.y[0], orb.z[0], '.', markersize=10, label='Initial Position', zorder=1)


        else:
            orb = orbits[0]
            path = ax.plot3D(orb.x[0], orb.y[0], orb.z[0], label=f'Trajectory', alpha=0.8, zorder=0)
            position = ax.plot3D(orb.x[0], orb.y[0], orb.z[0], '.', markersize=10, label='Initial Position', zorder=1)

        ax.legend()

    else:
        # Plot start of orbital path
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

        ax.legend()

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

    #plt.show()

    return fig, ax, ani


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


def gibbs(r1, r2, r3, mu, format='sv'):
    """
    This orbital determination approach requires 3 position measurements: r1, r2, r3.
    format = 'sv' will return the second position measurement r2 and its velocity v2.
    format = 'coe' will return a vector of the orbital elements.
    """
    # Define the magnitudes of each position measurement
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    r3_mag = np.linalg.norm(r3)

    # Make the measurements arrays if not already
    r1 = np.array(r1)
    r2 = np.array(r2)
    r3 = np.array(r3)

    # Calculate the S parameter
    S = r1 * (r2_mag - r3_mag) + r2 * (r3_mag - r1_mag) + r3 * (r1_mag - r2_mag)

    # Calculate the D parameter
    D = np.cross(r1, r2) + np.cross(r2, r3) + np.cross(r3, r1)

    # Calculate the N parameter
    N = r1_mag*(np.cross(r2, r3)) + r2_mag*(np.cross(r3, r1)) + r3_mag*(np.cross(r1, r2))

    # Calculate the velocity for position measurement 2
    v2 = np.sqrt(mu / (np.linalg.norm(N) * (np.linalg.norm(D)))) * (np.cross(D, r2) / r2_mag + S)

    if format == 'sv': # Return postion and velocity state vector
        return r2, v2
    
    elif format == 'coe': # Return classical orbital elements
        return coe_from_sv(r2, v2, mu)


def stumpff_C(z): # Stumpff function C(z)
    N = 8
    c = 0
    for k in range(N):
        c += (-1)**k * z**k / math.factorial(2*k + 2)
    return c


def stumpff_S(z): # Stumpff function S(z)
    N = 8
    s = 0
    for k in range(N):
        s += (-1)**k * z**k / math.factorial(2*k + 3)
    return s


def y(z, r1_mag, r2_mag, A):
    """
    Helper function for F, which is used as an abstraction layer in the solution to Lambert's problem.
    """
    y = r1_mag + r2_mag + A * (z * stumpff_S(z) - 1) / np.sqrt(stumpff_C(z))
    return y


def F(z, dt, r1_mag, r2_mag, A): 
    """
    This is a helper function for solving Lambert's Problem that incorporates stumpff functions.
    """
    global mu
    yy = y(z, r1_mag, r2_mag, A) 
    F = (yy / stumpff_C(z))**(3/2) * stumpff_S(z) + A * np.sqrt(yy) - np.sqrt(mu) * dt
    return F


def lambert(r1, r2, dt, mu, format='sv'):
    """
    This orbital determination approach requires 2 measured position vectors and a measured time between them.
    The solution 
    """
    # Define the magnitudes of each position measurement
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # Make the measurements arrays if not already
    r1 = np.array(r1)
    r2 = np.array(r2)

    # Assume a prograde trajectory
    cross = np.cross(r1, r2)
    arg = np.dot(r1,r2) / (r1_mag * r2_mag)
    
    # Calculate change in true anomaly
    if cross[2] >= 0:
        dtheta = np.arccos(arg)
    else:
        dtheta = 2*np.pi - np.arccos(arg) 
    
    # Calculate A variable
    A = np.sin(dtheta) * np.sqrt(r1_mag * r2_mag / (1 - np.cos(dtheta)))

    # Find zero-crossing of F to get a good initial condition
    z = 0
    while F(z, dt, r1_mag, r2_mag, A) < 0:
        z = z + 0.1

    # Use Newton's method to solve F for z
    sol = root(F, z, tol=1e-8, args=(dt, r1_mag, r2_mag, A))
    z_root = sol.x[0]

    # Calculate y for this root
    y_root = y(z_root, r1_mag, r2_mag, A)

    # Calculate lagrange coefficients
    f = 1 - y_root / r1_mag
    g = A * np.sqrt(y_root / mu)
    g_dot = 1 - y_root / r2_mag

    # Calculate v2
    v1 = 1/g * (r2 - f * r1)
    v2 = 1/g * (g_dot * r2 - r1)
    

    if format == 'sv': # Return postion and velocity state vector
        return r2, v2
    
    elif format == 'coe': # Return classical orbital elements
        return coe_from_sv(r2, v2, mu)
    


def gauss(R1, R2, R3, rho1_hat, rho2_hat, rho3_hat, t1, t2, t3, format='sv'):
    """
    This aproach requires 3 measurements of topocentric equatorial unit vectors to characterize an orbit.    
    """
    
    global mu

    # Turn into arrays if not already
    R1 = np.array(R1)
    R2 = np.array(R2)
    R3 = np.array(R3)
    rho1_hat = np.array(rho1_hat)
    rho2_hat = np.array(rho2_hat)
    rho3_hat = np.array(rho3_hat)

    # Turn R matrix into array if not already
    R = np.array([R1, R2, R3])

    # Calculate tau's
    tau1 = t1 - t2
    tau3 = t3 - t2
    tau = tau3 - tau1
    
    # Construct the P matrix
    p1 = np.cross(rho2_hat, rho3_hat)
    p2 = np.cross(rho1_hat, rho3_hat)
    p3 = np.cross(rho1_hat, rho2_hat)
    P = np.transpose(np.array([p1, p2, p3]))

    # Calculate the D matrix
    D0 = np.dot(rho1_hat, p1)
    if D0 == 0:
        D0 = 1e-10
    D = R @ P
    
    # Calculate A, B, and E parameters
    A = 1/D0 * (-D[0,1]*tau3/tau + D[1,1] + D[2,1]*tau1/tau)
    B = 1/(6*D0) * (D[0,1]*(tau3**2 - tau**2)*tau3/tau + D[2,1]*(tau**2-tau1**2)*tau1/tau)
    E = np.dot(R2, rho2_hat)

    # Calculate a, b, and c parameters
    a = -(A**2 + 2*A*E + np.linalg.norm(R2)**2)
    b = -2*mu*B*(A + E)
    c = -mu**2 * B**2

    # Solve for polynomial roots
    roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c])
    # Find min positive real roots
    x = np.max([np.real(x) for x in roots if np.abs(np.imag(x)) < 1e-8 and np.real(x) > 0])
    
    # Calculate lagrange coefficients
    f1 = 1 - 1/2*mu*tau1**2/x**3
    f3 = 1 - 1/2*mu*tau3**2/x**3
    g1 = tau1 - 1/6*mu*(tau1/x)**3
    g3 = tau3 - 1/6*mu*(tau3/x)**3

    # Calculate rhos
    rho1 = 1/D0 * ((6*(D[2,0]*tau1/tau3 + D[1,0]*tau/tau3)*x**3 \
            + mu*D[2,0]*(tau**2 - tau1**2)*tau1/tau3) \
            / (6*x**3 + mu*(tau**2 - tau3**2)) - D[0,0])
    
    rho2 = A + mu*B/x**3

    rho3 = 1/D0 * ((6*(D[0,2]*tau3/tau1 - D[1,2]*tau/tau1)*x**3 \
            + mu*D[0,2]*(tau**2 - tau3**2)*tau3/tau1) \
            /(6*x**3 + mu*(tau**2 - tau3**2)) - D[2,2])

    # Calculate position vectors
    r1 = R1 + rho1 * rho1_hat
    r2 = R2 + rho2 * rho2_hat
    r3 = R3 + rho3 * rho3_hat

    # Calvuate v2
    v2 = 1 / (f1*g3 - f3*g1) * (-f3*r1 + f1*r3)


    # Iterative improvement loop
    rho1_old = rho1
    rho2_old = rho2 
    rho3_old = rho3

    diff1 = 1 
    diff2 = 1 
    diff3 = 1
    n = 0
    nmax = 1000
    tol = 1.e-8
    
    while ((diff1 > tol) and (diff2 > tol) and \
           (diff3 > tol)) and (n < nmax):

        n = n+1
        
        # for universal kepler’s equation:
        ro = np.linalg.norm(r2)
        vo = np.linalg.norm(v2)
        vro = np.dot(v2,r2)/ro
        a = 2/ro - vo**2/mu
        x1 = kepler_U(tau1, ro, vro, a)
        x3 = kepler_U(tau3, ro, vro, a)
        
        # Calculate new lagrange coefficients
        ff1, gg1 = f_and_g(x1, tau1, ro, a)
        ff3, gg3 = f_and_g(x3, tau3, ro, a)
        
        # Update f and g with average
        f1 = (f1 + ff1)/2
        f3 = (f3 + ff3)/2
        g1 = (g1 + gg1)/2
        g3 = (g3 + gg3)/2
        
        
        # Get new rho
        c1 = g3/(f1*g3 - f3*g1)
        c3 = -g1/(f1*g3 - f3*g1)
        rho1 = 1/D0*( -D[0, 0] + 1/c1*D[1, 0] - c3/c1*D[2, 0])
        rho2 = 1/D0*( -c1*D[0, 1] + D[1, 1] - c3*D[2, 1])
        rho3 = 1/D0*(-c1/c3*D[0, 2] + 1/c3*D[1, 2] - D[2, 2])
        
        # Calculate new position vectors
        r1 = R1 + rho1 * rho1_hat
        r2 = R2 + rho2 * rho2_hat
        r3 = R3 + rho3 * rho3_hat
        
        # Calculate v2
        v2 = (-f3*r1 + f1*r3)/(f1*g3 - f3*g1)
        
        # Calculace difference from iteration
        diff1 = abs(rho1 - rho1_old)
        diff2 = abs(rho2 - rho2_old)
        diff3 = abs(rho3 - rho3_old)
        
        # Update rhos
        rho1_old = rho1
        rho2_old = rho2
        rho3_old = rho3
        

    if format == 'sv': # Return postion and velocity state vector
        return r2, v2
    
    elif format == 'coe': # Return classical orbital elements
        return coe_from_sv(r2, v2, mu)
    

def f_and_g(x, t, ro, a):

    global mu

    z = a*x**2
    f=1- x**2/ro*stumpff_C(z)
    g=t- 1/np.sqrt(mu)*x**3*stumpff_S(z)

    return f, g


def kepler_U(dt, ro, vro, a):
    global mu 

    #Starting value for x:
    x0 = np.sqrt(mu)*np.abs(a)*dt

    # Function to solve
    def f(x):
        z = a*x**2
        f = ro*vro/np.sqrt(mu) * x**2 * stumpff_C(z) + \
            (1 - a*ro) * x**3 * stumpff_S(z) + ro*x - np.sqrt(mu)*dt
        return f

    roots = root(f, x0)
    f_out = np.min([y for y in roots.x if np.imag(y) < 1e-8])

    return f_out


def observation(time, r_sat, latitude = 0):
    """
    This function calculates the position vector of an observer rotating with the Earth in the grocentric equatorial frame.
    Assuming observer is in the plane of the vernal equinox at t = 0.
    latitude is angle in radians.
    """

    global Re

    w = 2 * np.pi * 1 / (24*3600 - 4*60) # Sidereal rotation frequency
    alpha = w * time # Angle the observer has rotated from the vernal equinox

    xo = Re * np.cos(latitude) * np.cos(alpha)
    yo = Re * np.cos(latitude) * np.sin(alpha)
    zo = Re * np.sin(latitude)

    # Calculate position vector
    R =  [xo, yo, zo]

    # Calculate rho
    rho = r_sat - R
    rhohat = rho / np.linalg.norm(rho)

    return R, rho, list(rhohat)


def add_gaussian_noise(x, percent_std):
    std = percent_std * np.sqrt(np.sum(x**2) / len(x))
    noise = np.random.normal(0, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 


def print_orbital_elements(coe):
    names = ['h', 'e', 'i', 'omega', 'w']#, 'theta']
    for idx, name in enumerate(names):
        print(name + f': {coe[idx]}')

    return


if __name__ == "__main__":
    # Inputs
    Re = 6371e3 # m, Earth radius
    Me = 5.97e24 # kg, Earth mass
    G = 6.67e-11 # Gravitational constant

    # # Inputs
    DIMENSIONS = 3 # 2D sim right now
    latitude = 45 * np.pi / 180
    perigee = 400e3 # m
    i = 10 * np.pi / 180 # Inclin.
    w = 50 * np.pi / 180 # AoP
    omega = 10 * np.pi / 180 # RAAN
    solver = 'LSODA'
    reltol = 1e-14
    e = 0.1
    T = 1e5 # Time period over which to evaluate the numerical solution

    # Intermediates
    rp = Re + perigee
    mu = G * Me
    
    r0 = rp # m

    # Calculate angular momentum
    h = np.sqrt((e+1) * rp * mu)
    h_true = h

    # Print initial elements
    print('INPUT ORBITAL ELEMENTS:')
    input_elements = [h, e, i, omega, w]
    print_orbital_elements(input_elements)
    print('\n')

    r0, v0, _, _ = sv_from_coe(h, e, omega, i, w, 0, mu)

    rvec = np.array([1, 0, 0])                              
    rvec_0 = r0 * (rvec / np.linalg.norm(rvec))
    vvec = np.array([0, np.cos(i*np.pi/180), np.sin(i*np.pi/180)]) 
    vvec_0 = v0 * (vvec / np.linalg.norm(vvec))
    s0 = [*r0, *v0]
    # Calculating numerical solution
    # x_n, y_n, z_n, theta_n, time_n = numericalSolution(s0, mu, reltol, T, solver, oblate=obvec[idx], drag=drag[idx])
    orbit_n = numericalSolution(s0, mu, reltol, T, solver)   


    indices = [10, 30, 50]
    X = orbit_n.x[indices]
    Y = orbit_n.y[indices]
    Z = orbit_n.z[indices]
    t = orbit_n.t[indices]

    r = np.transpose(np.array([X, Y, Z]))

    # Generate observation and rho vectors for gauss estimation
    Ro0 = [Re*np.cos(latitude), 0, Re*np.sin(latitude)]
    R = []
    rho = []

    for idx, index in enumerate(indices):
        Ri, rhoi, rhohati = observation(t[idx], r[idx], latitude)
        R += [Ri]
        rho += [rhohati]

    # Minimum Measurements
    # Estimate orbit with Gibbs' method
    r2, v2 = gibbs(r[0], r[1], r[2], mu)
    s_gibbs = [*r2, *v2]
    orbit_gibbs = numericalSolution(s_gibbs, mu, reltol, T, solver)
    h, e, omega, i, w, theta = coe_from_sv(r2, v2, mu)

    print('GIBBS ESTIMATION:')
    output_elements_gibbs = [h, e, i, omega, w]
    print_orbital_elements(output_elements_gibbs)
    print('\n')

    # Estimate orbit with Lambert's method
    r2, v2 = lambert(r[0], r[2], t[2] - t[0], mu)
    s_lam = [*r2, *v2]
    orbit_lam = numericalSolution(s_lam, mu, reltol, T, solver)
    h, e, omega, i, w, theta = coe_from_sv(r2, v2, mu)

    print('LAMBERT ESTIMATION:')
    output_elements_lam = [h, e, i, omega, w]
    print_orbital_elements(output_elements_lam)
    print('\n')

    # Estimate orbit with gauss method
    r2, v2 = gauss(R[0], R[1], R[2], rho[0], rho[1], rho[2], t[0], t[1], t[2])
    s_gauss = [*r2, *v2]
    orbit_gauss = numericalSolution(s_gauss, mu, reltol, T, solver)
    h, e, omega, i, w, theta = coe_from_sv(r2, v2, mu)

    print('GAUSS ESTIMATION:')
    output_elements_gauss = [h, e, i, omega, w]
    print_orbital_elements(output_elements_gauss)
    print('\n')


    # Plot minimum estimation orbits
    solvec = [orbit_n, orbit_gibbs, orbit_lam, orbit_gauss]

    fig, ax, _ = plot_3D_orbits(solvec, Re)
    ax.scatter3D(X, Y, Z, '*', label='Measurements')
    ax.legend()
    
    # Plot error comparison
    fig, ax = plt.subplots()
    barWidth = 0.25
    xAxis = np.arange(5)
    gibbs_ratio = (np.array(output_elements_gibbs) / np.array(input_elements) - 1)*100
    lam_ratio = (np.array(output_elements_lam) / np.array(input_elements) - 1)*100
    gauss_ratio = (np.array(output_elements_gauss) / np.array(input_elements) - 1)*100
    ax.bar(xAxis - barWidth, gibbs_ratio, width = barWidth, label='Gibbs')
    ax.bar(xAxis, lam_ratio, width = barWidth, label='Lambert')
    ax.bar(xAxis + barWidth, gauss_ratio, width = barWidth, label='Gauss')
    # ax.set_xticklabels(['', 'h', 'e', 'i', 'RAAN', 'AoP'])
    ax.grid('enable')
    ax.legend()
    ax.set_title('Percent Error in Orbital Element Estimation')
    ax.set_ylabel('Percent Error [%]')


    # Noisy Identification    
    indices = [10, 30, 50]
    percent_error = 0.001
    X = add_gaussian_noise(orbit_n.x[indices], percent_error)
    Y = add_gaussian_noise(orbit_n.y[indices], percent_error)
    Z = add_gaussian_noise(orbit_n.z[indices], percent_error)
    t = add_gaussian_noise(orbit_n.t[indices], percent_error)

    # Plot noisy inputs
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(orbit_n.t, orbit_n.x, '--', label='x clean')
    ax[0].plot(orbit_n.t, orbit_n.y, '--', label='y clean')
    ax[0].plot(orbit_n.t, orbit_n.z, '--', label='z clean')

    ax[0].plot(orbit_n.t, add_gaussian_noise(orbit_n.x, percent_error), alpha=0.6, label='x noisy')
    ax[0].plot(orbit_n.t, add_gaussian_noise(orbit_n.y, percent_error), alpha=0.6, label='y noisy')
    ax[0].plot(orbit_n.t, add_gaussian_noise(orbit_n.z, percent_error), alpha=0.6, label='z noisy')

    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Position [m]')
    ax[0].set_title('Clean vs Noisy Orbital Position Measurements')
    ax[0].grid('enable')
    ax[0].legend()

    ax[1].plot(orbit_n.t, orbit_n.z - add_gaussian_noise(orbit_n.z, percent_error), alpha=0.6, label='z noise only')

    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Position [m]')
    ax[1].set_title('Z Noise')
    ax[1].grid('enable')
    ax[1].legend()

    fig.tight_layout()


    r = np.transpose(np.array([X, Y, Z]))

    # Estimate orbit with Gibbs' method
    r2, v2 = gibbs(r[0], r[1], r[2], mu)
    s_gibbs = [*r2, *v2]
    orbit_gibbs = numericalSolution(s_gibbs, mu, reltol, T, solver)
    h, e, omega, i, w, theta = coe_from_sv(r2, v2, mu)

    print('GIBBS ESTIMATION:')
    output_elements_gibbs = [h, e, i, omega, w]
    print_orbital_elements(output_elements_gibbs)
    print('\n')

    # Estimate orbit with Lambert's method
    r2, v2 = lambert(r[0], r[2], t[2] - t[0], mu)
    s_lam = [*r2, *v2]
    orbit_lam = numericalSolution(s_lam, mu, reltol, T, solver)
    h, e, omega, i, w, theta = coe_from_sv(r2, v2, mu)

    print('LAMBERT ESTIMATION:')
    output_elements_lam = [h, e, i, omega, w]
    print_orbital_elements(output_elements_lam)
    print('\n')

    # Estimate orbit with gauss method
    r2, v2 = gauss(R[0], R[1], R[2], rho[0], rho[1], rho[2], t[0], t[1], t[2])
    s_gauss = [*r2, *v2]
    orbit_gauss = numericalSolution(s_gauss, mu, reltol, T, solver)
    h, e, omega, i, w, theta = coe_from_sv(r2, v2, mu)

    print('GAUSS ESTIMATION:')
    output_elements_gauss = [h, e, i, omega, w]
    print_orbital_elements(output_elements_gauss)
    print('\n')


    # Plot minimum estimation orbits
    solvec = [orbit_n, orbit_gibbs, orbit_lam, orbit_gauss]

    fig, ax, _ = plot_3D_orbits(solvec, Re)
    ax.scatter3D(X, Y, Z, '*', label='Measurements')
    ax.legend()
    
    # Plot error comparison
    fig, ax = plt.subplots()
    barWidth = 0.25
    xAxis = np.arange(5)
    gibbs_ratio = np.abs((np.array(output_elements_gibbs) / np.array(input_elements) - 1)*100)
    lam_ratio = np.abs((np.array(output_elements_lam) / np.array(input_elements) - 1)*100)
    gauss_ratio = np.abs((np.array(output_elements_gauss) / np.array(input_elements) - 1)*100)
    ax.bar(xAxis - barWidth, gibbs_ratio, width = barWidth, label='Gibbs')
    ax.bar(xAxis, lam_ratio, width = barWidth, label='Lambert')
    ax.bar(xAxis + barWidth, gauss_ratio, width = barWidth, label='Gauss')
    ax.set_xticklabels(['', 'h', 'e', 'i', 'RAAN', 'AoP'])
    ax.grid('enable')
    ax.legend()
    ax.set_title('Percent Error in Orbital Element Estimation')
    ax.set_ylabel('Percent Error [%]')
    ax.set_yscale('log')

    # Multiple samplings of noisy data, lambert approach
    percent_error = 0.001
    x_noise = add_gaussian_noise(orbit_n.x, percent_error)
    y_noise = add_gaussian_noise(orbit_n.y, percent_error)
    z_noise = add_gaussian_noise(orbit_n.z, percent_error)
    t_noise = add_gaussian_noise(orbit_n.t, percent_error)
    
    # Create empty arrays of orbital elements
    numSamples = 2
    hvec = []
    hvec_mean = []
    evec = []
    evec_mean = []
    omegavec = []
    omegavec_mean = []
    ivec = []
    ivec_mean = []
    wvec = []
    wvec_mean = []

    for jdx in range(numSamples - 1):
        # Generate samples
        sampleIndices = np.array([jdx, jdx+1]) * 47 + 25

        X = x_noise[sampleIndices]
        Y = y_noise[sampleIndices]
        Z = z_noise[sampleIndices]
        t = t_noise[sampleIndices]
        r = np.transpose(np.array([X, Y, Z]))

        print(jdx)

        # Estimate orbit with Lambert's method
        r2, v2 = lambert(r[0], r[1], t[1] - t[0], mu)
        s_lam = [*r2, *v2]
        orbit_lam = numericalSolution(s_lam, mu, reltol, T, solver)
        h, e, omega, i, w, theta = coe_from_sv(r2, v2, mu)

        hvec.append(h)
        hvec_mean.append(np.mean(hvec))
        evec.append(e)
        evec_mean.append(np.mean(evec))
        omegavec.append(omega)
        omegavec_mean.append(np.mean(omegavec))
        ivec.append(i)
        ivec_mean.append(np.mean(ivec))
        wvec.append(w)
        wvec_mean.append(np.mean(wvec))
    
    # Plot convergence
    fig, ax = plt.subplots(5, 2)
    ax[0][0].plot(hvec)
    ax[0][0].axhline(input_elements[0])
    ax[0][0].set_xlabel('Sample Number')
    ax[0][0].set_title('Specific Angular Momentum [m**2/s]')
    ax[0][0].grid('enable')


    ax[0][1].plot(np.array(hvec_mean) / input_elements[0] * 100)
    ax[0][1].axhline(100)
    ax[0][1].set_xlabel('Sample Number')
    ax[0][1].set_title('Average Specific Angular Momentum [%]')
    ax[0][1].grid('enable')

    ax[1][0].plot(evec)
    ax[1][0].axhline(input_elements[1])
    ax[1][0].set_xlabel('Sample Number')
    ax[1][0].set_title('Eccentricity [-]')
    ax[1][0].grid('enable')

    ax[1][1].plot(np.array(evec_mean) / input_elements[1] * 100)
    ax[1][1].axhline(100)
    ax[1][1].set_xlabel('Sample Number')
    ax[1][1].set_title('Average Eccentricity [%]')
    ax[1][1].grid('enable')

    ax[2][0].plot(omegavec)
    ax[2][0].axhline(input_elements[2])
    ax[2][0].set_xlabel('Sample Number')
    ax[2][0].set_title('RAAN [rad]')
    ax[2][0].grid('enable')

    ax[2][1].plot(np.array(omegavec_mean) / input_elements[2] * 100)
    ax[2][1].axhline(100)
    ax[2][1].set_xlabel('Sample Number')
    ax[2][1].set_title('Average RAAN [%]')
    ax[2][1].grid('enable')
    
    ax[3][0].plot(ivec)
    ax[3][0].axhline(input_elements[3])
    ax[3][0].set_xlabel('Sample Number')
    ax[3][0].set_title('Inclination [rad]')
    ax[3][0].grid('enable')

    ax[3][1].plot(np.array(ivec_mean) / input_elements[3] * 100)
    ax[3][1].axhline(100)
    ax[3][1].set_xlabel('Sample Number')
    ax[3][1].set_title('Average Inclination [%]')
    ax[3][1].grid('enable')

    ax[4][0].plot(wvec)
    ax[4][0].axhline(input_elements[4])
    ax[4][0].set_xlabel('Sample Number')
    ax[4][0].set_title('AoP [rad]')
    ax[4][0].grid('enable')
    
    ax[4][1].plot(np.array(wvec_mean) % np.pi / input_elements[4] * 100)
    ax[4][1].axhline(100)
    ax[4][1].set_xlabel('Sample Number')
    ax[4][1].set_title('Average AoP [%]')
    ax[4][1].grid('enable')


    fig.tight_layout()

    plt.show()
    
