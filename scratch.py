#%%
import numpy as np


w0 = (5e-3)/2 # m
wavelength = 500e-9 # m
f = 1 # m

d =  f * np.pi**2 * w0**4 / (np.pi**2 * w0**4 + f**2 * wavelength**2)

print(1-d)


