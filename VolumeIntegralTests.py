#%%
import numpy as np
import matplotlib.pyplot as plt

Re = 6370e3
G = 6.67e-11
Me = 5.9e24

mu = G*Me

r = np.array([7000e3, 7000e3, 7000e3]) # Position

# Calculate acceleration
N = 10            
rho_0 = 4400 # kg/m**3
drho = 100
rvec = np.linspace(0, Re, N)
thetavec = np.linspace(0, np.pi, N)
phivec = np.linspace(0, 2*np.pi, N)

dr = rvec[1] - rvec[0]
dt = thetavec[1] - thetavec[0]
dq = phivec[1] - phivec[0]

a = np.zeros(3) # acceleration, m/s**2
for ri in rvec:
    for ti in thetavec:
        for qi in phivec:
            # Calculate cartesian coords of integration point
            xi = ri * np.sin(ti) * np.cos(qi)
            yi = ri * np.sin(ti) * np.sin(qi)
            zi = ri * np.cos(ti)
            
            # Calculate distance between orbiting body and integration point
            dx = r[0] - xi
            dy = r[1] - yi
            dz = r[2] - zi
            d = np.sqrt(dx**2 + dy**2 + dz**2)
            dhat = np.array([dx, dy, dz]) / d

            print(d)

            # Calculate incremental acceleration
            rho = rho_0 + drho * np.sin(ti) # Density at this location
            da = rho / d**2 * ri**2 * np.sin(ti) * dr * dt * dq
            a = a + da * dhat

# Account for -G
a *= -G    
print(len(a))     


sym_a = -mu/np.linalg.norm(r)**3 * r

print(f'Integrated Volume: {a}')
print(f'Analytical Volume: {sym_a}')