
from tkinter import * 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import cumtrapz

# plot function is created for 
# plotting the graph in 
# tkinter window
def plot():

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
    ax.plot(x, y, '--', color='0.7')
    ax.set_aspect('equal', 'box')


    # Plot orbiting body
    frameSpeed = 20 # ms btw frames
    speed = 1000 # Multiplier
    T = 0
    x_c = np.interp(T, time, x)
    y_c = np.interp(T, time, y)
    position, = ax.plot(x_c, y_c, 'o') 

    # Defining animation generator function
    def animate(i):
        T = (speed*i * 1000/frameSpeed) % time[-1]
        x_c = np.interp(T, time, x)
        y_c = np.interp(T, time, y)
        position.set_xdata(x_c)
        position.set_ydata(y_c)  # update the data.
        return position,

    # Animating plot
    ani = animation.FuncAnimation(
        fig, animate, interval=frameSpeed, repeat=True, blit=True, save_count=50)

  
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master = window)  
    canvas.draw()
  
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()
  
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()


if __name__ == "__main__":
    # the main Tkinter window
    window = Tk()
    
    # setting the title 
    window.title('Plotting in Tkinter')
    
    # dimensions of the main window
    window.geometry("500x500")
    
    # button that displays the plot
    plot_button = Button(master = window, 
                        command = plot,
                        height = 2, 
                        width = 10,
                        text = "Update Plot")

    # Text boxes for entry
    tb_perigee = Text(master = window,
                      height=1,
                      width=5)
    tb_perigee.pack()

    tb_initVelocity = Text(master = window,
                    height=1,
                    width=5)
    tb_initVelocity.pack()

    tb_earthMass = Text(master = window,
                    height=1,
                    width=5)
    tb_earthMass.pack()
    
    # place the button 
    # in main window
    plot_button.pack()
    
    # run the gui
    window.mainloop()