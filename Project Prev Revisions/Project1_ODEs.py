#%%
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt

# Define global variables
DIMENSIONS = 2 # 2D sim right now

# Inputs
Re = 6371e3 # m, Earth radius
Me = 5.97e24 # kg, Earth mass
G = 6.67e-11 # Gravitational constant
perigee = 1200e3 # m
v0 = 10000 # m/s

# Intermediates
rp = Re + perigee
h = rp * v0
mu = G * Me

# Numerical parameters
initial_state = [rp, 0, 0, v0]
method = 'RK45'
abstol = 1e-16

def state_derivative(time, state):
    r = state[:DIMENSIONS] # Position
    v = state[DIMENSIONS:] # Velocity
    a = - mu / np.linalg.norm(r)**3 * r
    return [*v, *a]

sol = sci.solve_ivp(state_derivative, [0, 1e6], initial_state, vectorized=True, rtol = abstol, atol=abstol)

# def solve_orbit(abstol):
#     solution = sci.solve_ivp(state_derivative, [0, 1e6], initial_state, vectorized=True, atol=abstol)
#     return solution

# for idx in range()

fig, ax = plt.subplots()
ax.plot(sol.y[0], sol.y[1], '--', color='0.5', alpha=0.3)
plot = ax.scatter(sol.y[0], sol.y[1], c=sol.t)
cbar = fig.colorbar(plot, label='Simulation Time [s]')
#fig.colorbar(sol.t, ax=ax)
plt.show()
