# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import integrate

# import plotly.graph_objects as go
# import numpy as np

# # Create figure
# fig = go.Figure()

# # Add traces, one for each slider step
# for step in np.arange(0, 5, 0.1):
#     fig.add_trace(
#         go.Scatter(
#             visible=False,
#             line=dict(color="#00CED1", width=6),
#             name="𝜈 = " + str(step),
#             x=step*np.cos(step * np.arange(0, 10, 0.01)),
#             y=np.sin(step * np.arange(0, 10, 0.01))))

# # Make 10th trace visible
# fig.data[10].visible = True

# # Create and add slider
# steps = []
# for i in range(len(fig.data)):
#     step = dict(
#         method="update",
#         args=[{"visible": [False] * len(fig.data)},
#               {"title": "Slider switched to step: " + str(i)}],  # layout attribute
#     )
#     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
#     steps.append(step)

# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "Frequency: "},
#     pad={"t": 50},
#     steps=steps
# )]

# fig.update_layout(
#     sliders=sliders
# )

# fig.show()

# # # Define functions




# # # Main
# # if __name__ == "__main__":

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import cumtrapz

# Inputs
Re = 6371e3 # m, Earth radius
Me = 5.97e24 # kg, Earth mass
G = 6.67e-11 # Gravitational constant
perigee = 1200e3 # m
v0 = 10000 # m/s

rp = Re + perigee
h = rp * v0
mu = G * Me

e = h**2/mu * 1/rp -1

thetavec = np.linspace(0, 2*np.pi, 1000)

# Calculate orbital positions
r = h**2/mu * 1/(1 + e*np.cos(thetavec))
x = r * np.cos(thetavec)
y = r * np.sin(thetavec)

# Calculate time
time = [0] + list(h**3 / mu**2 * cumtrapz(1/(1 + e*np.cos(thetavec))**2))

# Plot orbit
fig, ax = plt.subplots()

# Plot Earth
x_earth = Re * np.cos(thetavec)
y_earth = Re * np.sin(thetavec)

ax.plot(x_earth, y_earth, color='teal')
split = int(len(x_earth)/2)
ax.fill_between(x_earth[:split], y_earth[:split], y_earth[split:], color='teal', alpha=0.5)

# Plot orbit path

#ax.scatter(x, y, s=5, c=thetavec, cmap='magma')
ax.plot(x, y, '--', color='0.7')
ax.set_aspect('equal', 'box')


# Plot orbiting body
frameSpeed = 20 # ms btw frames
speed = 1000 # Multiplier
T = 0
x_c = np.interp(T, time, x)
y_c = np.interp(T, time, y)
position, = ax.plot(x_c, y_c, 'o') 

def animate(i):
    T = (speed*i * 1000/frameSpeed) % time[-1]
    x_c = np.interp(T, time, x)
    y_c = np.interp(T, time, y)
    position.set_xdata(x_c)
    position.set_ydata(y_c)  # update the data.
    return position,


ani = animation.FuncAnimation(
    fig, animate, interval=frameSpeed, repeat=True, blit=True, save_count=50)

plt.show()