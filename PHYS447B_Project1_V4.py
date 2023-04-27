#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import LogLocator
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
def exactSolution(e, rp, mu, n=1000, theta=None):
    # Define true anomaly vector
    if theta == None:
        theta = np.linspace(0, 2*np.pi, n)
    else:
        theta = [theta]

    # Calculate angular momentum
    h = np.sqrt((e+1) * rp * mu)

    # Calculate orbital positions
    r = h**2/mu * 1/(1 + e*np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Calculate time
    if len(np.array(theta)) > 1:
        time = [0] + list(h**3 / mu**2 * sci.cumtrapz(1/(1 + e*np.cos(theta))**2, theta))
    else:
        time = 0

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

    sol = sci.solve_ivp(state_derivative, [0, T], initial_state, method=solver, vectorized=True, rtol=reltol, atol=reltol)
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
fig.set_size_inches(9, 10)

fig2, ax2 = plt.subplots(3, 2) # Plotting time series
fig2.set_size_inches(8, 8)

# Define lists to store solutions
e_sol = [[],[],[],[]]
n_sol = [[],[],[],[]]
solv_opts = [[],[],[]]

# Iterating through eccentricities
for idx, e in enumerate(evec):
    x_e, y_e, theta_e, time_e = exactSolution(e, rp, mu, n=N) # Calculate the exact solution
    e_sol[idx] = Orbit(x_e, y_e, theta_e, time_e) # Store the exact solution

    # Calculate plot index
    col = idx % 2
    row = int(2 * np.floor(idx/2))

    # Plot Earth
    x_earth = Re * np.cos(theta_e) * 1e-3
    y_earth = Re * np.sin(theta_e) * 1e-3
    ax[row][col].plot(x_earth, y_earth, color='teal')
    split = int(len(x_earth)/2)
    ax[row][col].fill_between(x_earth[:split], y_earth[:split], y_earth[split:], color='teal', alpha=0.5)

    # Plot exact orbit path
    ax[row][col].plot(x_e * 1e-3, y_e * 1e-3, color='k', label='Exact Solution')
    ax[row][col].set_aspect('equal', 'box')
    ax[row][col].set_title(f'Orbital Path, e = {e}')
    ax[row][col].set_xlabel('X Position [km]')
    ax[row][col].set_ylabel('Y Position [km]')

    n_sol[idx] = len(solver_list)*[[]]

    # Iterating through solvers
    for jdx, solver in enumerate(solver_list):
        t_0 = time.time() # Time before solver call
        x_n, y_n, theta_n, time_n = numericalSolution(e, rp, mu, reltol, T, solver)
        t_1 = time.time() # Time after solver call
        solutionTime = t_1 - t_0

        print(f"e = {e}, solver = {solver}, simtime = {solutionTime} s")

        # Calculate error at each numerical solution
        error = np.zeros(len(theta_n))
        prevTh = 0
        for kdx, Th in enumerate(theta_n):
            if Th - prevTh < 0:
                break # Only calculate for the first orbit
            else:
                x_exact, y_exact, _, _ = exactSolution(e, rp, mu, theta=Th) # Exact x, y solution at this theta
                x_approx = x_n[kdx] # Numerical soln for x
                y_approx = y_n[kdx] # Numerical soln for y
                error[kdx] = np.sqrt((x_exact - x_approx)**2 + (y_exact - y_approx)**2)
            
            prevTh = Th


        # Store numerical solution
        n_sol[idx][jdx] = Orbit(x_n, y_n, theta_n, time_n, solver, error, solutionTime)
    
        # Plot numerical orbit path
        ax[row][col].plot(x_n*1e-3, y_n*1e-3)

        # Plot numerical error
        theta_d = theta_n*180/np.pi
        cmap_dict = {'RK45': 'winter', 'LSODA': 'Wistia'}
        ax[row+1][col].semilogy(theta_d, error, '.-', label=f'Numerical Solution: {solver}', alpha=0.3)
        ax[row+1][col].set_ylim([1e-15, 1e-2])
        

        # Label error plot
        ax[row+1][col].set_xlabel('True Anomaly [deg]')
        ax[row+1][col].set_ylabel('Error [m]')
        ax[row+1][col].set_title('Error between Numerical and Exact Solns')
        ax[row+1][col].legend()

        # A bunch of garbage to get goddamn minor gridlines
        ax[row+1][col].grid('Enable')
        ax[row+1][col].minorticks_on()        
        ax[row+1][col].yaxis.set_major_locator(LogLocator(numticks=5))
        ax[row+1][col].yaxis.set_minor_locator(LogLocator(numticks=15,subs=1**(-np.arange(-15,-2))))
        ax[row+1][col].tick_params(axis='y', which='minor', labelleft=False) 
        ax[row+1][col].grid(visible=True, which='minor', color='0.9', linestyle='-')
        

    # Correct plotting problems for parabolic case
    if e >= 1: 
        # Limit axes in extreme cases
        ax[row][col].set_xlim([-10*Re*1e-3, 5*Re*1e-3])
        ax[row][col].set_ylim([-5*Re*1e-3, 5*Re*1e-3])


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
fig4, ax4 = plt.subplots(4, 2)
fig4.set_size_inches(9, 10)
long_sol = len(solver_list)*[[]]
e = 0.3

# Calculate exact orbit
x_e, y_e, theta_e, time_e = exactSolution(e, rp, mu, N) # Calculate the exact solution
exact_sol = Orbit(x_e, y_e, theta_e, time_e) # Store the exact solution

for idx, solver in enumerate(solver_list):
    # Calculate numerical orbit
    solver = solver_list[idx]
    x_n, y_n, theta_n, time_n = numericalSolution(e, rp, mu, reltol, T*5, solver)
    long_sol[idx] = Orbit(x_n, y_n, theta_n, time_n, solver)

    # Convert time to hrs
    time_days = np.array(time_n) / 3600 / 24



    # Plot exact orbit path
    delta = 20000
    ax4[1][idx].plot(x_e, y_e, color='k', label='Exact Solution')
    ax4[1][idx].set_aspect('equal', 'box')
    ax4[1][idx].set_ylim([-2*delta, delta])
    ax4[1][idx].set_xlim([min(x_n) - delta, min(x_n) + delta])
    ax4[1][idx].set_xlabel('\nX Position [m]')
    ax4[1][idx].set_ylabel('Y Position [m]')
    ax4[1][idx].set_title('Zoomed In on Apogee')
    ax4[1][idx].set_anchor('C')

    # Plot numerical orbit path
    plot1 = ax4[0][idx].scatter(x_n * 1e-3, y_n * 1e-3, c=time_days, cmap='cool', alpha=0.5)
    plot2 = ax4[1][idx].scatter(x_n, y_n, c=time_days, cmap='cool', alpha=0.5)
    ax4[1][idx].set_aspect('equal', 'box')
    cbar = fig4.colorbar(plot2, ax=ax4[1][idx])
    cbar.ax.set_ylabel('Time [days]')

    # Plot Earth
    x_earth = Re * np.cos(theta_e) * 1e-3
    y_earth = Re * np.sin(theta_e) * 1e-3
    ax4[0][idx].plot(x_earth, y_earth, color='teal')
    split = int(len(x_earth)/2)
    ax4[0][idx].fill_between(x_earth[:split], y_earth[:split], y_earth[split:], color='teal', alpha=0.5)
    ax4[0][idx].set_xlabel('\nX Position [km]')
    ax4[0][idx].set_ylabel('Y Position [km]')
    ax4[0][idx].set_title(f'Orbital Path, e = {e},\nSolver = {solver}')
    ax4[0][idx].set_aspect('equal', 'box')
    ax4[0][idx].set_anchor('C')
    cbar = fig4.colorbar(plot1, ax=ax4[0][idx])
    cbar.ax.set_ylabel('Time [days]')

    # Plot timestep histogram
    ax4[2][idx].hist(np.diff(time_n), bins=30, facecolor='tomato', alpha=0.7, edgecolor='tomato')
    ax4[2][idx].set_title(f'Histogram of Timesteps\nSolver = {solver}')
    ax4[2][idx].set_xlabel('Timestep [s]')
    ax4[2][idx].grid('enable', alpha=0.5)
    ax4[2][idx].set_anchor('C')


    # Calculate error at each numerical solution
    error = np.zeros(len(theta_n))
    for kdx, Th in enumerate(theta_n):
        x_exact, y_exact, _, _ = exactSolution(e, rp, mu, theta=Th) # Exact x, y solution at this theta
        x_approx = x_n[kdx] # Numerical soln for x
        y_approx = y_n[kdx] # Numerical soln for y
        error[kdx] = np.sqrt((x_exact - x_approx)**2 + (y_exact - y_approx)**2)
    
    # Plot error
    theta_d = np.unwrap(theta_n) / 2 / np.pi # normalize to number of orbits
    ax4[3][idx].semilogy(theta_d,error)
    ax4[3][idx].set_title(f'Error vs Orbit Number\nSolver = {solver}')
    ax4[3][idx].set_xlabel('Orbit Number [-]')
    ax4[3][idx].set_ylabel('Error [m]')
    ax4[3][idx].set_ylim([1e-9, 1e-1])
    ax4[3][idx].grid('enable', alpha=0.5)
    ax4[3][idx].set_anchor('C')

    # Add subticks
    ax[3][idx].grid('Enable')
    ax[3][idx].minorticks_on()        
    ax[3][idx].yaxis.set_major_locator(LogLocator(numticks=3))
    ax[3][idx].yaxis.set_minor_locator(LogLocator(numticks=15,subs=1**(-np.arange(-9,-1))))
    ax[3][idx].tick_params(axis='y', which='minor', labelleft=False) 
    ax[3][idx].grid(visible=True, which='minor', color='0.9', linestyle='-')
    


fig4.tight_layout()
plt.show()

