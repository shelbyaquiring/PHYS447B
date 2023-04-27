#%%
import numpy as np
import matplotlib.pyplot as plt
import spiceTools as st
import spiceypy as spice
import lambert as lm
import scipy as sp
import scipy.integrate as sci
from scipy.optimize import curve_fit



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

# Function for fitting a plane to orbits
def planeFit(data, a, b, c):
        x = data[0]
        y = data[1]
        return a*x + b*y + c

# Skew function for vector rotations
def skew(s):
        return np.array([[0,    -s[2],  s[1]], \
                        [s[2],    0,  -s[0]], \
                        [-s[1], s[0],    0 ]])

if __name__ == "__main__":
        
    # Inputs
    Re = 6371e3 # m, Earth radius
    ae = 300e3 # m altitude, Earth parking orbit
    Rv = 6052e3 # m, Venus radius
    av = 300e3 # m altitude, Venus parking orbit

    Me = 5.97e24 # kg, Earth mass
    Ms = 1.99e30 # kg, Sun mass
    Mv = 4.867e24 # kg, venus mass
    
    G = 6.67e-11 # Gravitational constant
    mu = 132712e15 # Sun mu
    mu_e = 398199e9 # Earth mu
    mu_v = 324900e9 # Venus mu

    g0 = 9.80665 # Earth surface gravity, m/s^2
    Isp_aj = 320 # s, specific impulse
    Isp_rl = 450 # s, specific impulse
    mf = 20 # kg, vehicle dry mass

    AU = 1.496e8 * 1e3 # km to AU scaling factor

    FRAME='ECLIPJ2000'
    OBSERVER='SUN'
    STEPS = 36545

    # Get orbital data from NASA SPICE files
    spice.furnsh("SPICE\solar_system_kernel.mk")
    filename = "SPICE\de432s.bsp"
    ids, names, tcs_sec, tcs_cal = st.get_objects(filename)

    times = st.tc2array(tcs_sec[0],STEPS) # Generate time vector
    rs = [] # Generate empty vetor for orbital data

    xlims = [np.min(times), np.min(times[200:])]
    plane_params = [] # List for storing fitted plane parameters
    idx = 0

    # Generate figure for plotting orbits
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    r = 1.5
    N = 100
    X, Y = np.meshgrid(np.linspace(-r, r, N), np.linspace(-r, r, N))

    # Extract orbital data for earth and venus
    for name in names:
        if 'BARYCENTER' in name and ('EARTH' in name or 'VENUS' in name):
            r = st.get_ephemeris_data(name, times, FRAME, OBSERVER) * 1e3
            rs.append(r)

            # Plane fit
            pos = rs[-1]
            ax.plot(pos[:,0] / AU, pos[:,1] / AU, pos[:,2] / AU, label=name)
            params, cov = curve_fit(planeFit, [pos[:,0] / AU, pos[:,1] / AU], pos[:,2] / AU)
            plane_params.append(params)
            ax.plot_surface(X, Y, planeFit([X, Y], *params), alpha=0.3)
            idx += 1

    # Calculate plane vectors based on plane fit
    c1 = [*plane_params[0][:2], 1]
    c2 = [*plane_params[1][:2], 1]
    v1 = np.transpose([[0, 0, 0], c1]) / 10
    v2 = np.transpose([[0, 0, 0], c2]) / 10
    c3 = np.cross(c1, c2)

    plane_angle = np.arccos(np.dot(c1, c2) / np.linalg.norm(c1) / np.linalg.norm(c2))
    print(f'Angle between Orbital Planes: {plane_angle * 180/np.pi} [deg]')
    C3 = c3 / np.linalg.norm(c3)

    v3 = np.transpose([[0, 0, 0], c3]) * 10
    # ax.plot(*v1)
    # ax.plot(*v2)
    # ax.plot(*v3)

    # PORKCHOP PLOT ##################################
    startidx_0 = 26801 # Initial start date
    endidx_0 = startidx_0 # Initial arrival date
    span = 1000 # Range of start dates, days
    step = 10 # Increments to test, days

    startdate = spice.timout(times[startidx_0], "YYYY MMM DD HR:MN:SC.### (TDB) ::TDB")
    print(f'Start Date:\n{startdate}')
    enddate = spice.timout(times[endidx_0], "YYYY MMM DD HR:MN:SC.### (TDB) ::TDB")
    print(f'End Date:\n{enddate}')

    # Generate grid to iterate over
    dx_vals = np.arange(365*3, step=step*3)
    dy_vals = np.arange(365, step=step)
    X, Y = np.meshgrid(dx_vals, dy_vals)

    # Initialize array for storing Delta V values
    deltaV = np.zeros((len(dy_vals), len(dx_vals)))

    for idx, xinc in enumerate(dx_vals):
        for jdx, yinc in enumerate(dy_vals):
            
            startidx = startidx_0 + xinc
            # endidx = endidx_0 + yinc 
            endidx = startidx + yinc

            index = [startidx, endidx]
            DT = times[index[1]] - times[index[0]]

            # Extract the position and velocity of earth and venus at the desired mission start and end times
            RE = np.array([x[:3] for x in rs[1][index]])
            RV = np.array([x[:3] for x in rs[0][index]])
            VE = np.array([x[3:] for x in rs[1][index]])
            VV = np.array([x[3:] for x in rs[0][index]])

            # TEMP: Make z vals 0 for coplanar assumption
            RE[:,2] = np.array([0,0])
            RV[:,2] = np.array([0,0])
            VE[:,2] = np.array([0,0])
            VV[:,2] = np.array([0,0])

            R1 = RE[0] # Earth position at initial time, m
            R2 = RV[1] # Venus position at final time, m

            # Calculate transfer orbit solution, prograde
            r1_pro, v1_pro, r2_pro, v2_pro = lm.lambert(R1, R2, DT, mu, direction='prograde')
            r1_ret, v1_ret, r2_ret, v2_ret = lm.lambert(R1, R2, DT, mu, direction='retrograde')

            # # Calculate dV at departure and arrival for prograde and retrograde solutions
            dV1_pro = np.linalg.norm(VE[0] - v1_pro)
            dV2_pro = np.linalg.norm(VV[1] - v2_pro)
            dV1_ret = np.linalg.norm(VE[0] - v1_ret)
            dV2_ret = np.linalg.norm(VV[1] - v2_ret)

            dV1 = np.min([dV1_pro, dV1_ret])
            dV2 = np.min([dV2_pro, dV2_ret])

            dV = dV1 + dV2
            deltaV[jdx][idx] = dV

    # Plot dV surface
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, deltaV*1e-3, alpha=0.6, cmap='cool')
    ax.set_title('Earth-Venus Total Delta V')
    ax.set_xlabel('Departure Days since May 1st 2023 [days]')
    ax.set_ylabel('Flight Duration [days]')
    ax.set_zlabel('Delta V [km/s]')

    # Plot 2D porkchops
    fig, ax = plt.subplots()
    indices = deltaV*1e-3 < 15 # Threshold dV values
    p = ax.scatter(X[indices], Y[indices], c=deltaV[indices]*1e-3, cmap='cool')
    ax.contour(X, Y, deltaV*1e-3)
    c = fig.colorbar(p)
    c.set_label('Total Delta V [km/s]')

    # Find minimum dV
    deltaV_clean = np.array([x for y in np.transpose(deltaV) for x in y if not np.isnan(x)])
    min_index = deltaV == np.nanmin(deltaV_clean)
    opt = ax.scatter(X[min_index], Y[min_index], s=50, color='red', label='Minimum dV')

    # Add lines for each year
    yearlines = np.arange(1,4)*365 - 120
    for idx, year in enumerate(yearlines):
        if idx == 0:
            label='Jan 1st Lines'
        else:
            label=''
        ax.axvline(year, color='0.5', alpha=0.5, linestyle='--', label=label)

    # Add plot labels
    ax.legend()
    ax.set_title('Earth-Venus Total Delta V')
    ax.set_xlabel('Departure Days since May 1st 2023 [days]')
    ax.set_ylabel('Flight Duration [days]')

    # Plot dV vs departure date for 150 day transit time
    fix, ax = plt.subplots(2,1)
    yloc = Y == 150
    ax[0].plot(X[yloc], deltaV[yloc]*1e-3)
    ax[0].grid('enable')
    ax[0].set_title('Earth-Venus Total Delta V, Contour at 150 Days Transit Time')
    ax[0].set_xlabel('Departure Days since May 1st 2023 [days]')
    ax[0].set_ylabel('Delta V for 150 day Transit TIme [km/s]')

    ax[1].plot(X[yloc], deltaV[yloc]*1e-3)
    ax[1].grid('enable')
    ax[1].set_title('Earth-Venus Total Delta V, Zoomed In on Mimima')
    ax[1].set_xlabel('Departure Days since May 1st 2023 [days]')
    ax[1].set_ylabel('Delta V for 150 day Transit TIme [km/s]')
    ax[1].set_ylim([4, 7])
    fig.tight_layout()

    # Plot the optimized transfer orbit
    testpoint = np.squeeze([X[min_index],Y[min_index]])
    startidx = startidx_0 + testpoint[0]
    endidx = startidx + testpoint[1]

    index = [startidx, endidx]
    DT = times[index[1]] - times[index[0]]

    RE = np.array([x[:3] for x in rs[1][index]])
    RV = np.array([x[:3] for x in rs[0][index]])
    VE = np.array([x[3:] for x in rs[1][index]])
    VV = np.array([x[3:] for x in rs[0][index]])

    # TEMP: Make z vals 0 for RV
    RE[:,2] = np.array([0,0])
    RV[:,2] = np.array([0,0])
    VE[:,2] = np.array([0,0])
    VV[:,2] = np.array([0,0])

    R1 = RE[0] # Earth position at initial time, m
    R2 = RV[1] # Venus position at final time, m

    # Calculate transfer orbit solution, prograde
    r1_pro, v1_pro, r2_pro, v2_pro = lm.lambert(R1, R2, DT, mu, direction='prograde')
    r1_ret, v1_ret, r2_ret, v2_ret = lm.lambert(R1, R2, DT, mu, direction='retrograde')

    # Propogate transf er orbit solution
    orbit_p = numericalSolution([*r1_pro, *v1_pro], T=DT)
    orbit_r = numericalSolution([*r1_ret, *v1_ret], T=DT)

    # Calculate intersection with plane change axis
    intersection = np.abs(orbit_p.y - C3[1]/C3[0] * orbit_p.x)/AU
    fig, ax = plt.subplots()
    ax.plot(orbit_p.t, intersection)
    index_axis = intersection == np.min(intersection)


    # Print the departure and arrival velocities of the transfer orbit
    print(f'Vd: {np.linalg.norm(v1_pro)*1e-3} [km/s]')
    print(f'Va: {np.linalg.norm(v2_pro)*1e-3} [km/s]')

    # Calculate v_inf for departure and arrival
    V_inf_E = v1_pro - VE[0]
    V_inf_V = v2_pro - VV[1]

    print(f'V_E at Departure: {np.linalg.norm(VE[0]) * 1e-3} [km/s]')
    print(f'V_V at Arrival: {np.linalg.norm(VV[1]) * 1e-3} [km/s]')

    print(f'V_infty_E: {np.linalg.norm(V_inf_E) * 1e-3} [km/s]')
    print(f'V_infty_V: {np.linalg.norm(V_inf_V) * 1e-3} [km/s]')

    # Calculate the average radii from sun to each planet
    Rse = np.mean([np.linalg.norm(x[:3]) for x in rs[1]]) # Avg radius from sun to earth
    Rsv = np.mean([np.linalg.norm(x[:3]) for x in rs[0]]) # Avg radius from sun to venus

    # Calculate the sphere of influence radius
    R_SOI_E = Rse * (Me/Ms)**(2/5)
    R_SOI_V = Rsv * (Mv/Ms)**(2/5)

    print(f'SOIe: {R_SOI_E / Re} [Re]')
    print(f'SOIe: {R_SOI_E * 1e-3 * 1e-5} [10^5 km]')

    print(f'SOIv: {R_SOI_V / Rv} [Rv]')
    print(f'SOIv: {R_SOI_V * 1e-3 * 1e-5} [10^5 km]')


    # Calculate dV
    dV1_pro = np.linalg.norm(VE[0] - v1_pro)
    dV2_pro = np.linalg.norm(VV[1] - v2_pro)
    dV1_ret = np.linalg.norm(VE[0] - v1_ret)
    dV2_ret = np.linalg.norm(VV[1] - v2_ret)


    dV1 = np.min([dV1_pro, dV1_ret])
    dV2 = np.min([dV2_pro, dV2_ret])

    dV_pro = dV1_pro + dV2_pro
    dV_ret = dV1_ret + dV2_ret

    AU = 1.496e8 * 1e3 # km to AU

    fig, ax = plt.subplots()
    ax.plot(rs[0][:,0] / AU, rs[0][:,1] / AU, alpha=0.5, label='Venus Orbit')
    ax.plot(rs[1][:,0] / AU, rs[1][:,1] / AU, alpha=0.5, label='Earth Orbit')
    ax.scatter(RV[:,0] / AU, RV[:,1] / AU, label='Venus')
    ax.scatter(RE[:,0] / AU, RE[:,1] / AU, label='Earth')
    ax.plot(orbit_p.x / AU, orbit_p.y / AU, label='Transfer Orbit') #label=f'Prograde, dV = {np.round(dV_pro*1e-3,2)} km/s'
    ax.plot([-C3[0], C3[0]], [-C3[1], C3[1]], alpha=0.5, label='Plane Intersection')
    ax.scatter(orbit_p.x[index_axis] / AU, orbit_p.y[index_axis] / AU, marker='o', label='Plane Change')

    # ax.plot(orbit_r.x, orbit_r.y, label=f'Retrograde, dV = {np.round(dV_ret*1e-3,2)} km/s')
    ax.set_title('Transfer Orbit between Non-Coplanar Orbits')
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.grid('enable')
    ax.set_aspect('equal')
    ax.legend()


    # Calculate velocity of circular parking orbits
    vc_e = np.sqrt(mu_e / (Re + ae))
    vc_v = np.sqrt(mu_v / (Rv + av))

    print(f'vc_e = {vc_e}')
    print(f'vc_v = {vc_v}')

    # Calculate delta V to achieve v_inf
    dV_e = vc_e * (np.sqrt(2 + (np.linalg.norm(V_inf_E)/vc_e)**2) - 1)
    dV_v = vc_v * (np.sqrt(2 + (np.linalg.norm(V_inf_V)/vc_v)**2) - 1)

    print(f'dV_e: {dV_e}')
    print(f'dV_v: {dV_v}')
    print(f'dV: {dV_v + dV_e}')


    # Plane change maneuver
    s_cross = skew(C3)
    v_before = np.array([orbit_p.vx[index_axis], orbit_p.vy[index_axis], orbit_p.vz[index_axis]])
    v_after = sp.linalg.expm(plane_angle * s_cross) @ v_before
    print(f'Before: {v_before}')
    print(f'After: {v_after}')

    # Calculate delta V from plane change
    dV_planeChange = np.linalg.norm(v_before - v_after)
    print(f'dV_planeChange: {dV_planeChange}')

    # Calculate total delta V
    dV_total_3 = dV_v + dV_e + dV_planeChange
    print(f'Mission dV: {np.round(dV_total_3*1e-3,2)} [km/s]')

    # dV
    dV_aj = dV_v + dV_planeChange
    dV_rl = dV_e

    # Calculate wet mass
    m1 = mf * np.exp(dV_aj / (g0 * Isp_aj))
    mo = m1 * np.exp(dV_rl / (g0 * Isp_rl))

    print(f'Initial Mass: {m1} [kg]')
    print(f'Propelant Mass Fraction: {(m1 - mf)/m1 * 100} %')

    print(f'Initial Mass: {mo} [kg]')
    print(f'Propelant Mass Fraction, Total Vehicle: {(mo - m1)/mo * 100} %')

    # Show plots
    plt.show()

# %%
