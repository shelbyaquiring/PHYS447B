#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

m = 1
b = 0.1
k = 1

initial_state = [10, 2]

T = 10
solver = "LSODA"
tol = 1e-12
timevec = np.linspace(0, T, 1000)

def stateDerivativeIVP(time, state):
    x = state[0]
    xd = state[1]
    xdd = (-b*xd -k*x) / m
    return [xd, xdd]

def stateDerivativeBVP(time, state):
    print(np.shape(state[0]))
    x = state[0]
    xd = state[1]
    xdd = (-b*xd -k*x) / m
    return [xd, xdd]

def bc(ya, yb):
    out = np.array([ya[0] - Ya[0], yb[0] - Yb[0]])
    return out

sol_ivp = sci.solve_ivp(stateDerivativeIVP, [0, T], y0=initial_state, rtol=tol, solver=solver)

yout = np.ones((2, timevec.size))
Ya = sol_ivp.y[:,0]
Yb = sol_ivp.y[:,-1]
sol_bvp = sci.solve_bvp(stateDerivativeBVP, bc, timevec, yout, tol=tol)


fig, ax = plt.subplots()
ax.plot(sol_ivp.t, sol_ivp.y[0], label='Position, IVP')
ax.plot(sol_ivp.t, sol_ivp.y[1], label='Velocity, IVP')

ax.plot(sol_bvp.x, sol_bvp.y[0], '--', label='Position, BVP')
ax.plot(sol_bvp.x, sol_bvp.y[1], '--', label='Velocity, BVP')

ax.grid('enable')
ax.legend()

