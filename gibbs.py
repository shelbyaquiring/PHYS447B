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
    idx3 = 700
    ax.scatter([X[idx1], X[idx2], X[idx3]], [Y[idx1], Y[idx2], Y[idx3]]) # Plot them
    R1 = [X[idx1], Y[idx1], 0]
    R2 = [X[idx2], Y[idx2], 0]
    R3 = [X[idx3], Y[idx3], 0]
    DT = time[idx2] - time[idx1]
    t1 = thetavec[idx1]
    t2 = thetavec[idx2]


    # Estimate orbit
    
    r2, v2 = gibbs(R1, R2, R3, mu)

    h, e, omega, i, w, theta = coe_from_sv(r2, v2, mu)
    ax.scatter(np.linalg.norm(r2) * np.cos(theta), np.linalg.norm(r2) * np.sin(theta))
    
    output_elements = [h, e, i, omega, w]
    print_orbital_elements(output_elements)



