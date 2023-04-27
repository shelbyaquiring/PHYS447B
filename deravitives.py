#%%
import numpy as np
from sympy import *


# Calulate partial derivatives
x, y, z, r, mu, J2, Re = symbols('x y z r mu J2 Re', real=True)

r = (x**2 + y**2 + z**2)**0.5

Vg = J2*mu*Re**2/r**3* (1/2*(3*(z/r)**2 - 1))

#differntiating function Vg in respect to x, y, z
ax = diff(Vg, x)
ay = diff(Vg, y)
az = diff(Vg, z)

ax2 = ax.cancel()#ax.

#ax.as_expr()
#ay.as_expr()
#az.as_expr()

display(ax.cancel().simplify())
display(ay.cancel().simplify())
display(az.cancel())
# %%
