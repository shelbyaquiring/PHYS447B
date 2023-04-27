#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px

def multipole(phi, theta):
    s = - 1/4 * np.sqrt(5/np.pi) * (3*np.cos(theta)**2 - 1)
    return s

def plot_3D_orbit(rvec, R):
    # Figure setup
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', computed_zorder=False)

    # Draw sphere
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
    color = multipole(u, v)
    x = R * np.cos(u) * np.sin(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(v)
    # ax.plot_surface(x, y, z, cmap="Blues", alpha=0.5, edgecolors='tab:blue', zorder=0)
    ax.plot_surface(x, y, z, facecolors=plt.cm.magma(color), alpha=0.85, edgecolors='tab:blue', zorder=0)
    # ax.contour3D(x, y, z, cmap="Blues", alpha=1, zorder=0)
    # # ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), cmap="Blues")
    #ax.plot_wireframe(x, y, z, color='steelblue', alpha=0.82, zorder=0)

    # Plot orbital path
    ax.plot3D(rvec[0], rvec[1], rvec[2], label='Trajectory', zorder=0)
    ax.plot3D(rvec[0][0], rvec[1][0], rvec[2][0], '.', label='Initial Position', zorder=1)

    # Add axis vectors
    l = 2*R
    start = [0,0,0]
    x,y,z = [start,start,start]
    u,v,w = [[l,0,0],[0,l,0],[0,0,l]]
    ax.quiver(x,y,z,u,v,w, color='k', alpha=0.5, arrow_length_ratio=0.1)
    
    # Add labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    ax.legend()

    return fig, ax




    


    

    # px, py, pz = rvec
    # fig2 = go.Figure(data=[go.Surface(x=x, y=y, z=z)])

    # fig2.update_layout(title='Mt Bruno Elevation', autosize=False,
    #                 width=500, height=500,
    #                 margin=dict(l=65, r=50, b=65, t=90))

    # fig2.add_scatter3d(x=px, y=py, z=pz)

    # fig2.show()

    


if __name__ == "__main__":
    # Inputs
    Re = 6371e3 # m, Earth radius
    Me = 5.97e24 # kg, Earth mass
    G = 6.67e-11 # Gravitational constant

    # Plotting
    thetavec = np.linspace(0,np.pi*2,1000)
    x = 1.1*Re * np.cos(thetavec)
    y = 1.1*Re * np.sin(thetavec)
    z = 1.1*Re * np.zeros_like(x)
    position = [x,y,z]
    plot_3D_orbit(position, Re)

    plt.show()

