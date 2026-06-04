import numpy as np
from scipy.integrate import solve_ivp
import math

v = 246.22
Mh = 230.0
Mt = 50.0
lambda0 = Mh**2 / (2*v**2)
yt0 = 0.93690 + 0.00556 * (Mt - 173.34)
g1 = 0.46
g2 = 0.65
g3 = 1.1666

def beta(t, y):
    g1, g2, g3, yt, lam = y
    dg1 = g1**3 / (16*math.pi**2) * (41.0/10)
    dg2 = g2**3 / (16*math.pi**2) * (-19.0/6)
    dg3 = g3**3 / (16*math.pi**2) * (-7.0)
    dyt = yt / (16*math.pi**2) * (9/2*yt**2 - 8*g3**2 - 9/4*g2**2 - 17/20*g1**2)
    dlam = 1/(16*math.pi**2) * (24*lam**2 + 12*lam*yt**2 - 6*yt**4 - 9*lam*g2**2 - 9/5*lam*g1**2 + 9/8*g2**4 + 27/200*g1**4 + 9/20*g2**2*g1**2)
    return [dg1, dg2, dg3, dyt, dlam]

t0 = 2 * math.log(172.5)
tPlanck = 2 * math.log(1.22e19)

sol = solve_ivp(beta, [t0, tPlanck], [g1, g2, g3, yt0, lambda0], method='RK45')

print(f"Final t: {sol.t[-1]}, max g1: {max(sol.y[0])}, max yt: {max(sol.y[3])}, max lam: {max(sol.y[4])}")
