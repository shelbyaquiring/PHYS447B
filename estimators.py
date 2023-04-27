import numpy as np
from scipy.optimize import root
import math


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


def F(z, dt, r1_mag, r2_mag, A, mu): 
    """
    This is a helper function for solving Lambert's Problem that incorporates stumpff functions.
    """
    # global mu

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
    while F(z, dt, r1_mag, r2_mag, A, mu) < 0:
        z = z + 0.1

    # Use Newton's method to solve F for z
    sol = root(F, z, tol=1e-8, args=(dt, r1_mag, r2_mag, A, mu))
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
    
    if format == 'sv_both': # Return postion and velocity state vector
        return r1, v1, r2, v2
    
    elif format == 'coe': # Return classical orbital elements
        return coe_from_sv(r2, v2, mu)
    


def gauss(R1, R2, R3, rho1_hat, rho2_hat, rho3_hat, t1, t2, t3, mu, format='sv'):
    """
    This aproach requires 3 measurements of topocentric equatorial unit vectors to characterize an orbit.    
    """
    
    # global mu

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
        x1 = kepler_U(tau1, ro, vro, a, mu)
        x3 = kepler_U(tau3, ro, vro, a, mu)
        
        # Calculate new lagrange coefficients
        ff1, gg1 = f_and_g(x1, tau1, ro, a, mu)
        ff3, gg3 = f_and_g(x3, tau3, ro, a, mu)
        
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
    

def f_and_g(x, t, ro, a, mu):

    # global mu

    z = a*x**2
    f=1- x**2/ro*stumpff_C(z)
    g=t- 1/np.sqrt(mu)*x**3*stumpff_S(z)

    return f, g


def kepler_U(dt, ro, vro, a, mu):
    # global mu 

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
