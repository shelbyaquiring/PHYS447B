#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as sci
from scipy.interpolate import interp1d
import time


# Inputs
DIMENSIONS = 2 # 2D sim right now
Re = 6371e3 # m, Earth radius
Me = 5.97e24 # kg, Earth mass
G = 6.67e-11 # Gravitational constant
perigee = 1000e3 # m
evec = [0, 0.3, 0.7, 1] # e = h**2/mu * 1/rp -1
solver_list = ['RK45',  'LSODA'] #  'Radau', 'BDF',
N = 1000 # Number of points over which to evaluate the exact solution
T = 5e4 # Time period over which to evaluate the numerical solution
reltol = 2.220446049250313e-14 # Relative tolerance for numerical solution

# Intermediates
rp = Re + perigee
mu = G * Me

# Define the exact solution aproach
def exactSolution(e, rp, mu, n):
    # Define true anomaly vector
    theta = np.linspace(0, 2*np.pi, n)

    # Calculate angular momntum
    h = np.sqrt((e+1) * rp * mu)

    # Calculate orbital positions
    r = h**2/mu * 1/(1 + e*np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Calculate time
    time = [0] + list(h**3 / mu**2 * sci.cumtrapz(1/(1 + e*np.cos(theta))**2, theta))

    return x, y, theta, time

# Define the numerical solution approach
def numericalSolution(e, rp, mu, reltol, T, solver):
    # Calculate angular momntum
    h = np.sqrt((e+1) * rp * mu)

    # Calculate initial velocity
    v0 = h / rp
    
    # Numerical parameters
    initial_state = [rp, 0, 0, v0]

    def state_derivative(time, state):
        r = state[:DIMENSIONS] # Position
        v = state[DIMENSIONS:] # Velocity
        a = - (mu / np.linalg.norm(r)**3) * r
        return [*v, *a]

    sol = sci.solve_ivp(state_derivative, [0, T], initial_state, method=solver, vectorized=True, rtol=reltol)
    theta = np.arctan2(sol.y[1], sol.y[0]) # Calculate true anomaly

    # Correct negative true anomaly
    for idx, angle in enumerate(theta):
        if angle < 0:
            theta[idx] += np.pi*2

    return sol.y[0], sol.y[1], theta, sol.t

# Class definition for storing an orbit solution
class Orbit:
    def __init__(self, x, y, theta, t, solver=None, error=None, soltime=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.theta = np.array(theta)
        self.t = np.array(t)
        self.solver = solver
        self.error = error
        self.soltime = soltime       

        # Calculate derivatives
        self.tdot = self.t[:-1] # Time vector minus the final element
        self.xdot = np.diff(self.x)/np.diff(self.t)
        self.ydot = np.diff(self.y)/np.diff(self.t)
        self.vmag = np.sqrt(self.xdot**2 + self.ydot**2)

# Set up plots
# Plot orbit
fig, ax = plt.subplots(4,2)
fig.set_size_inches(9, 9)

fig2, ax2 = plt.subplots(3, 2) # Plotting time series
fig2.set_size_inches(8, 8)

# Define lists to store solutions
e_sol = [[],[],[],[]]
n_sol = [[],[],[],[]]
solv_opts = [[],[],[]]

# Iterating through eccentricities
for idx, e in enumerate(evec):
    x_e, y_e, theta_e, time_e = exactSolution(e, rp, mu, N) # Calculate the exact solution
    e_sol[idx] = Orbit(x_e, y_e, theta_e, time_e) # Store the exact solution

    # Calculate plot index
    col = idx % 2
    row = int(2 * np.floor(idx/2))

    # Plot Earth
    x_earth = Re * np.cos(theta_e)
    y_earth = Re * np.sin(theta_e)
    ax[row][col].plot(x_earth, y_earth, color='teal')
    split = int(len(x_earth)/2)
    ax[row][col].fill_between(x_earth[:split], y_earth[:split], y_earth[split:], color='teal', alpha=0.5)

    # Plot exact orbit path
    ax[row][col].plot(x_e, y_e, color='k', label='Exact Solution')
    ax[row][col].set_aspect('equal', 'box')
    ax[row][col].set_title(f'Orbital Path, e = {e}\n')

    n_sol[idx] = len(solver_list)*[[]]

    # Iterating through solvers
    for jdx, solver in enumerate(solver_list):
        t_0 = time.time() # Time before solver call
        x_n, y_n, theta_n, time_n = numericalSolution(e, rp, mu, reltol, T, solver)
        t_1 = time.time() # Time after solver call
        solutionTime = t_1 - t_0

        print(f"e = {e}, solver = {solver}, simtime = {solutionTime} s")

        # Use true anomaly vector to interpolate exact and approximate orbits
        error = np.zeros(len(theta_e))

        for kdx, Th in enumerate(theta_e):
            x_exact = x_e[kdx] # Exact x solution at this theta
            y_exact = y_e[kdx] # Exact y solution at this theta
            fx = interp1d(theta_n, x_n, kind='cubic', fill_value='extrapolate')
            fy = interp1d(theta_n, y_n, kind='cubic', fill_value='extrapolate')
            x_approx = fx(Th) # Interpolated numerical soln for x
            y_approx = fy(Th) # Interpolated numerical soln for y
            error[kdx] = np.sqrt((x_exact - x_approx)**2 + (y_exact - y_approx)**2)

        # Store numerical solution
        n_sol[idx][jdx] = Orbit(x_n, y_n, theta_n, time_n, solver, error, solutionTime)
    
        # Plot numerical orbit path
        ax[row][col].plot(x_n, y_n)

        # Plot numerical error
        theta_d = theta_e*180/np.pi
        ax[row+1][col].semilogy(theta_d, error, '.-', label=f'Numerical Solution: {solver}', alpha=0.3)

        # Label error plot
        ax[row+1][col].set_xlabel('True Anomaly [deg]')
        ax[row+1][col].set_ylabel('Error [m]')
        ax[row+1][col].set_title('Relative Error between Orbital Aproximations\n')
        ax[row+1][col].grid('Enable')
        ax[row+1][col].legend()

    # Correct plotting problems for parabolic case
    if e >= 1: 
        # Limit axes in extreme cases
        ax[row][col].set_xlim([-10*Re, 5*Re])
        ax[row][col].set_ylim([-5*Re, 5*Re])
        ax[row+1][col].set_ylim([0, 1])


    # Plot positions and velocities of exact solution vs time
    if e < 1:
        ax2[0][0].plot(e_sol[idx].t / 3600, e_sol[idx].theta * 180/np.pi, '--', label=f'e = {e}')
        ax2[0][0].set_xlabel('Time [hrs]')
        ax2[0][0].set_ylabel('True Anomaly [deg]')
        ax2[0][0].set_title('True Anomaly vs Time')
        ax2[0][0].grid('enable')
        ax2[0][0].legend()

        ax2[0][1].plot(e_sol[idx].x * 1e-3, e_sol[idx].y * 1e-3, '--', label=f'e = {e}')
        ax2[0][1].set_xlabel('X Position [km]')
        ax2[0][1].set_ylabel('X Position [km]')
        ax2[0][1].set_title('X vs Y Position')
        ax2[0][1].grid('enable')
        ax2[0][1].set_aspect('equal', 'box')

        ax2[1][0].plot(e_sol[idx].t / 3600, e_sol[idx].x * 1e-3, '--', label=f'e = {e}')
        ax2[1][0].set_xlabel('Time [hrs]')
        ax2[1][0].set_ylabel('X Position [km]')
        ax2[1][0].set_title('X Position vs Time')
        ax2[1][0].grid('enable')

        ax2[1][1].plot(e_sol[idx].t / 3600, e_sol[idx].y * 1e-3, '--', label=f'e = {e}')
        ax2[1][1].set_xlabel('Time [hrs]')
        ax2[1][1].set_ylabel('Y Position [km]')
        ax2[1][1].set_title('Y Position vs Time')
        ax2[1][1].grid('enable')

        ax2[2][0].plot(e_sol[idx].tdot / 3600, e_sol[idx].xdot * 1e-3, '--', label=f'e = {e}')
        ax2[2][0].set_xlabel('Time [hrs]')
        ax2[2][0].set_ylabel('X Velocity [km/s]')
        ax2[2][0].set_title('X Velocity vs Time')
        ax2[2][0].grid('enable')

        ax2[2][1].plot(e_sol[idx].tdot / 3600, e_sol[idx].ydot * 1e-3, '--', label=f'e = {e}')
        ax2[2][1].set_xlabel('Time [hrs]')
        ax2[2][1].set_ylabel('Y Velocity [km/s]')
        ax2[2][1].set_title('Y Velocity vs Time')
        ax2[2][1].grid('enable')
        
fig.tight_layout()
fig2.tight_layout()

#%% Plot the times for each solver
fig3, ax3 = plt.subplots()
barWidth = 0.25
for idx, solver in enumerate(solver_list):
    bars = [x[idx].soltime for x in n_sol]
    xaxis = np.arange(len(evec))  + idx * barWidth
    print(xaxis)
    ax3.bar(xaxis, bars, width = barWidth, label=f"{solver}")

ax3.legend()
ax3.set_title('Solution Time Comparison Between ODE Solvers')
ax3.set_ylabel('Time [s]')
ax3.set_xlabel('Eccentricity [-]')
ax3.set_xticks([r + barWidth /len(solver_list) for r in range(len(evec))],
        [str(e) for e in evec])
ax3.grid('enable', alpha=0.3)


#%% Numerical divergence check
fig4, ax4 = plt.subplots()
long_sol = len(solver_list)*[[]]
e = 0.3

# Calculate exact orbit
x_e, y_e, theta_e, time_e = exactSolution(e, rp, mu, N) # Calculate the exact solution
exact_sol = Orbit(x_e, y_e, theta_e, time_e) # Store the exact solution

# Plot Earth
x_earth = Re * np.cos(theta_e)
y_earth = Re * np.sin(theta_e)
ax4.plot(x_earth, y_earth, color='teal')
split = int(len(x_earth)/2)
ax4.fill_between(x_earth[:split], y_earth[:split], y_earth[split:], color='teal', alpha=0.5)

# Plot exact orbit path
ax4.plot(x_e, y_e, color='k', label='Exact Solution')
ax4.set_aspect('equal', 'box')
ax4.set_title(f'Orbital Path, e = {e}\n')

solver = solver_list[1]
x_n, y_n, theta_n, time_n = numericalSolution(e, rp, mu, reltol, T*100, solver)
long_sol[idx] = Orbit(x_n, y_n, theta_n, time_n, solver)

# Plot numerical orbit path
plot = ax4.scatter(x_n, y_n, c=time_n, cmap='cool')
cbar = fig4.colorbar(plot)
cbar.ax.set_ylabel('Time [s]')

plt.show()

