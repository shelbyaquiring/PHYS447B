#%% File for testing an implemented solution to Lambert's problem
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, root
from scipy.integrate import cumtrapz
import math


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



def lambert(r1, r2, dt, mu, direction='prograde', format='sv'):
    """
    This orbital determination approach requires 2 measured position vectors and a measured time between them.
    The solution 
    """

    def F(z, dt, r1_mag, r2_mag, A): 
        """
        This is a helper function for solving Lambert's Problem that incorporates stumpff functions.
        """
        # global mu
        yy = y(z, r1_mag, r2_mag, A) 
        F = (yy / stumpff_C(z))**(3/2) * stumpff_S(z) + A * np.sqrt(yy) - np.sqrt(mu) * dt
        return F

    # Define the magnitudes of each position measurement
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # Make the measurements arrays if not already
    r1 = np.array(r1)
    r2 = np.array(r2)

    # Calculate change in angle
    cross = np.cross(r1, r2)
    arg = np.dot(r1,r2) / (r1_mag * r2_mag)
    dtheta = np.arccos(arg) # Calculate change in true anomaly

    # Differentiate prograde vs retrograde trajectories
    if direction == 'prograde':
        if cross[2] < 0:
            dtheta = 2*np.pi - dtheta
    elif direction == 'retrograde':
        if cross[2] > 0:
            dtheta = 2*np.pi - dtheta

    # Calculate A variable
    A = np.sin(dtheta) * np.sqrt(r1_mag * r2_mag / (1 - np.cos(dtheta)))

    # if direction == 'retrograde':
    #     A *= -1

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
        return r1, v1, r2, v2
    
    elif format == 'coe': # Return classical orbital elements
        return coe_from_sv(r2, v2, mu)
    


def print_orbital_elements(coe):
    names = ['h', 'e', 'i', 'omega', 'w']#, 'theta']
    for idx, name in enumerate(names):
        print(name + f': {coe[idx]}')

    return

if __name__ == '__main__':
    # Define orbital elements to test
    Re = 6371e3 # m, Earth radius
    Me = 5.97e24 # kg, Earth mass
    G = 6.67e-11 # Gravitational constant

    # Inputs
    perigee = 800e3 # m
    i = 0 * np.pi / 180 # Inclin.
    w = 0 # AoP
    omega = 0 # RAAN
    e = 0.3

    # Intermediates
    rp = Re + perigee
    mu = G * Me

    # Calculate angular momentum
    h = np.sqrt((e+1) * rp * mu)

    # Calculate vp
    vp = h / rp

    # Display input elements
    input_elements = [h, e, i, omega, w]
    print('INPUTS:')
    print_orbital_elements(input_elements)

    # Define vector of true anomaly
    thetavec = np.linspace(0, 2*np.pi, 1000)

    # Calculate orbital positions
    r = h**2/mu * 1/(1 + e*np.cos(thetavec))
    X = r * np.cos(thetavec)
    Y = r * np.sin(thetavec)

    # Calculate time
    time = [0] + list(h**3 / mu**2 * cumtrapz(1/(1 + e*np.cos(thetavec))**2, thetavec))

    # Plot orbit
    fig, ax = plt.subplots()

    # Plot Earth
    x_earth = Re * np.cos(thetavec)
    y_earth = Re * np.sin(thetavec)
    ax.plot(x_earth, y_earth, color='teal')
    split = int(len(x_earth)/2)
    ax.fill_between(x_earth[:split], y_earth[:split], y_earth[split:], color='teal', alpha=0.5)

    # Plot orbit path
    ax.plot(X, Y, '--', color='0.7')
    ax.set_aspect('equal', 'box')

    # Extract 2 positions
    idx1 = 100
    idx2 = 600
    ax.scatter([X[idx1], X[idx2]], [Y[idx1], Y[idx2]]) # Plot them
    R1 = [X[idx1], Y[idx1], 0]
    R2 = [X[idx2], Y[idx2], 0]
    DT = time[idx2] - time[idx1]
    t1 = thetavec[idx1]
    t2 = thetavec[idx2]


    # Estimate orbit
    
    r1, v1, r2, v2 = lambert(R1, R2, DT, mu)

    h, e, omega, i, w, theta = coe_from_sv(r1, v1, mu)
    ax.scatter(np.linalg.norm(r1) * np.cos(theta), np.linalg.norm(r1) * np.sin(theta))

    h, e, omega, i, w, theta = coe_from_sv(r2, v2, mu)
    ax.scatter(np.linalg.norm(r2) * np.cos(theta), np.linalg.norm(r2) * np.sin(theta))
    
    output_elements = [h, e, i, omega, w]
    print_orbital_elements(output_elements)



