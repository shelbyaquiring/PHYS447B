#%%
import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_objects(filename, display=True):
    objects = spice.spkobj(filename)

    ids, names, tcs_sec, tcs_cal = [], [], [], []
    n=0

    for o in objects:
        ids.append(o) # Add id

        # Get time coverage
        tc_sec = spice.wnfetd(spice.spkcov(filename, ids[n]), n)

        # Convert time coverage to human readable
        tc_cal=[spice.timout(f, "YYYY MDN DD HR:MN:SC.### (TDB) ::TDB") for f in tc_sec]

        # Append values
        tcs_sec.append(tc_sec)
        tcs_cal.append(tc_cal)

        # Get body name
        try:
            names.append(spice.bodc2n(o))
            if display:
                print(spice.bodc2n(o))
        except:
            names.append("Unknown Name")

    return ids, names, tcs_sec, tcs_cal

def tc2array(tcs,steps):
    out = np.zeros((steps,1))
    out[:,0]=np.linspace(tcs[0],tcs[1],steps)
    return out

def get_ephemeris_data(target, times, frame, observer):
    return np.array(spice.spkezr(target, times, frame, 'NONE',observer)[0])

def planeFit(data, a, b, c):
    x = data[0]
    y = data[1]
    return a*x + b*y + c

if __name__ == "__main__":
    FRAME='ECLIPJ2000'
    OBSERVER='SUN'
    STEPS = 10000

    spice.furnsh("SPICE\solar_system_kernel.mk")

    filename = "SPICE\de432s.bsp"

    ids, names, tcs_sec, tcs_cal = get_objects(filename)

    times = tc2array(tcs_sec[0],STEPS)
    rs = []

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    AU = 1.496e8


    r = 1.5
    N = 100
    X, Y = np.meshgrid(np.linspace(-r, r, N), np.linspace(-r, r, N))
    colors = ['red', 'blue']
    idx = 0

    for name in names:
        if 'BARYCENTER' in name and ('EARTH' in name or 'VENUS' in name):
            r=get_ephemeris_data(name, times, FRAME, OBSERVER)
            rs.append(r)
            pos = rs[-1]
            ax.plot(pos[:,0] / AU, pos[:,1] / AU, pos[:,2] / AU, label=name)

            # Plane fit
            params, cov = curve_fit(planeFit, [pos[:,0] / AU, pos[:,1] / AU], pos[:,2] / AU)
            ax.plot_surface(X, Y, planeFit([X, Y], *params), color=colors[idx], alpha=0.3)
            idx += 1

    # print(np.shape(rs[0]))
    ax.legend()
    # L = 5e9 / AU # Limit
    L = 2e8 / AU # Limit
    ax.set_xlim([-L,L])
    ax.set_ylim([-L,L])
    ax.set_zlim([-L/10,L/10])
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    ax.set_title('Planetary Orbits in Heliocentric Ecliptic Frame')
    plt.show()

# %%
