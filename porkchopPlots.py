#%%
import numpy as np
import matplotlib.pyplot as plt
import spiceTools as st
import spiceypy as spice
import lambert as lm
import scipy.integrate as sci

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


def numericalSolution(initial_state, T=1e3, reltol=1e-8, solver='RK45'):

    global mu

    # Define the state derivative function
    def state_derivative(time, state):
        # Extract position and velocity vectors
        r = state[:3] # Position

        if not np.linalg.norm(r) > Re: # If orbit is below the surface of the earth, no need to solve
            v = [0, 0, 0]
            a = [0, 0, 0]

        else:
            v = state[3:] # Velocity            
            a = - (mu / np.linalg.norm(r)**3) * r # Calculate the ideal 2-body monopole term
           
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


# Inputs
Re = 6371e3 # m, Earth radius
Me = 5.97e24 # kg, Earth mass
Ms = 1.99e30 # kg, Sun mass
G = 6.67e-11 # Gravitational constant
mu = 132712e15 # Sun mu

FRAME='ECLIPJ2000'
OBSERVER='SUN'
STEPS = 10000

spice.furnsh("SPICE\solar_system_kernel.mk")
filename = "SPICE\de432s.bsp"
ids, names, tcs_sec, tcs_cal = st.get_objects(filename)

times = st.tc2array(tcs_sec[0],STEPS)
rs = []

AU = 1.496e8

index = [0, 51]
DT = times[index[1]] - times[index[0]]

# r = 1.5
# N = 100
# X, Y = np.meshgrid(np.linspace(-r, r, N), np.linspace(-r, r, N))
# colors = ['red', 'blue']
# idx = 0

xlims = [np.min(times), np.min(times[200:])]

for name in names:
    if 'BARYCENTER' in name and ('EARTH' in name or 'VENUS' in name):
        r = st.get_ephemeris_data(name, times, FRAME, OBSERVER) * 1e3
        rs.append(r)

# r *= 1e3 # Convert km to m

RE = np.array([x[:3] for x in rs[1][index]])
RV = np.array([x[:3] for x in rs[0][index]])
VE = np.array([x[3:] for x in rs[1][index]])
VV = np.array([x[3:] for x in rs[0][index]])

# TEMP: Make z vals 0 for RV
RV[:,2] = np.array([0,0])

print(RV)

print(np.shape(RE))
print(np.shape(RV))

R1 = RE[0] # Earth position at initial time, m
R2 = RV[1] # Venus position at final time, m

# Calculate transfer orbit solution
r1, v1, r2, v2 = lm.lambert(R1, R2, DT, mu)


print(f'r1: {r1} [m]')
print(f'R1: {R1} [m]')
print(f'r2: {r2} [m]')
print(f'R2: {R2} [m]')
print(f'v1: {v1} [m/s]')
print(f'V1: {VE} [m/s]')
print(f'v2: {v1} [m/s]')
print(f'V2: {VV} [m/s]')


# Propogate transfer orbit solution
orbit_n = numericalSolution([*r1, *v1], T=DT)

fig, ax = plt.subplots()
ax.plot(rs[0][:,0], rs[0][:,1], alpha=0.5, label='Venus Orbit')
ax.plot(rs[1][:,0], rs[1][:,1], alpha=0.5, label='Earth Orbit')
ax.scatter(RV[:,0], RV[:,1], label='Venus')
ax.scatter(RE[:,0], RE[:,1], label='Earth')
ax.plot(orbit_n.x, orbit_n.y)



fig, ax = plt.subplots()
ax.plot(times, rs[0][:,0] / AU, label='Venus X')
ax.plot(times, rs[0][:,1] / AU, label='Venus Y')
ax.plot(times, rs[0][:,2] / AU, label='Venus Z')

ax.plot(times, rs[1][:,0] / AU, label='Earth X')
ax.plot(times, rs[1][:,1] / AU, label='Earth y')
ax.plot(times, rs[1][:,2] / AU, label='Earth Z')

ax.set_xlim(xlims)
ax.legend()

fig, ax = plt.subplots()
distance = np.sqrt((rs[0][:,0] - rs[1][:,0])**2 + (rs[0][:,1] - rs[1][:,1])**2 + (rs[0][:,2] - rs[1][:,2])**2)
ax.plot(times, distance / AU, label='Distance')
ax.scatter(times[index], distance[index] / AU, marker='o')
ax.set_xlim(xlims)
ax.legend()

plt.show()