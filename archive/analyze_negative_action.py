"""
Analysis of the point Mt=100 GeV, Mh=5 GeV:
  - Running of lambda(mu) and lambda_eff(mu) from Mt to M_Planck
  - Effective potential V_eff(phi) at high scales
  - Potential near the EW vev well (log-log up to 10^4 GeV)
  - Bounce action decomposition

Uses the same 3-loop RGEs and NNLO matching as the C++ solvers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

# ─── Constants ────────────────────────────────────────────────────────────────
v       = 246.22          # Higgs vev [GeV]
M_Pl    = 1.22e19         # Planck mass [GeV]
alpha3  = 0.1184
pi      = np.pi
PI2     = pi**2
LOOP1   = 16.0 * PI2
LOOP2   = LOOP1**2
LOOP3   = LOOP2 * LOOP1
LOOP4   = LOOP2**2

# ─── NNLO Matching ────────────────────────────────────────────────────────────
def get_nnlo_matching(Mh, Mt):
    dAlphas   = (alpha3 - 0.1184) / 0.0007
    lam_tree  = Mh**2 / (2.0 * v**2)
    yt_tree   = np.sqrt(2.0) * Mt / v
    d_lambda  = -0.00313 - 0.00004 * (Mt - 173.34)
    d_yt      = -0.0587 - 0.00042 * dAlphas
    lam_Mt    = lam_tree + d_lambda
    yt_Mt     = yt_tree + d_yt
    g1_Mt     = 0.46266 + 0.00006 * (Mt - 173.34)
    g2_Mt     = 0.65355 + 0.00002 * (Mt - 173.34)
    g3_Mt     = 1.1666  + 0.00314 * dAlphas - 0.00046 * (Mt - 173.34)
    Mb, Mtau  = 4.0, 1.777
    yb_Mt     = np.sqrt(2.0) * Mb  / v
    ytau_Mt   = np.sqrt(2.0) * Mtau / v
    return np.array([g1_Mt, g2_Mt, g3_Mt, yt_Mt, yb_Mt, ytau_Mt, lam_Mt])

# ─── Helper: unpack ──────────────────────────────────────────────────────────
def _u(y):
    g1, g2, g3, yt, yb, ytau, lam = y
    g1_2, g2_2, g3_2 = g1**2, g2**2, g3**2
    g1_4, g2_4, g3_4 = g1_2**2, g2_2**2, g3_2**2
    yt2, yb2, ytau2 = yt**2, yb**2, ytau**2
    yt4, yb4, ytau4 = yt2**2, yb2**2, ytau2**2
    return (g1,g2,g3,yt,yb,ytau,lam,
            g1_2,g2_2,g3_2, g1_4,g2_4,g3_4,
            yt2,yb2,ytau2, yt4,yb4,ytau4)

# ─── 3-loop beta functions ───────────────────────────────────────────────────
def beta_g1sq(y):
    (g1,g2,g3,yt,yb,ytau,lam,
     g1_2,g2_2,g3_2, g1_4,g2_4,g3_4,
     yt2,yb2,ytau2, yt4,yb4,ytau4) = _u(y)
    t1 = g1_4/LOOP1 * 41.0/10
    t2 = g1_4/LOOP2 * (44*g3_2/5 + 27*g2_2/10 + 199*g1_2/50
                        - 17*yt2/10 - yb2/2 - 3*ytau2/2)
    t3 = g1_4/LOOP3 * (
        yt2*(189*yt2/16 - 29*g3_2/5 - 471*g2_2/32 - 2827*g1_2/800)
      + lam*(-9*lam/5 + 9*g2_2/10 + 27*g1_2/50)
      + 297*g3_4/5 + 789*g2_4/64 - 388613*g1_4/24000
      - 3*g3_2*g2_2/5 - 137*g3_2*g1_2/75 + 123*g2_2*g1_2/160)
    return t1+t2+t3

def beta_g2sq(y):
    (g1,g2,g3,yt,yb,ytau,lam,
     g1_2,g2_2,g3_2, g1_4,g2_4,g3_4,
     yt2,yb2,ytau2, yt4,yb4,ytau4) = _u(y)
    t1 = g2_4/LOOP1 * (-19.0/6)
    t2 = g2_4/LOOP2 * (12*g3_2 + 35*g2_2/6 + 9*g1_2/10
                        - 3*yt2/2 - 3*yb2/2 - ytau2/2)
    t3 = g2_4/LOOP3 * (
        yt2*(147*yt2/16 - 7*g3_2 - 729*g2_2/32 - 593*g1_2/160)
      + lam*(-3*lam + 3*g2_2/2 + 3*g1_2/10)
      + 81*g3_4 + 324953*g2_4/1728 - 5597*g1_4/1600
      + 39*g3_2*g2_2 - g3_2*g1_2/5 + 873*g2_2*g1_2/160)
    return t1+t2+t3

def beta_g3sq(y):
    (g1,g2,g3,yt,yb,ytau,lam,
     g1_2,g2_2,g3_2, g1_4,g2_4,g3_4,
     yt2,yb2,ytau2, yt4,yb4,ytau4) = _u(y)
    g3_8  = g3_4**2; g3_10 = g3_8 * g3_2
    t1 = -g3_4/LOOP1 * 7
    t2 = g3_4/LOOP2 * (-26*g3_2 + 9*g2_2/2 + 11*g1_2/10 - 2*yt2 - 2*yb2)
    t3 = g3_4/LOOP3 * (
        yt2*(15*yt2 - 40*g3_2 - 93*g2_2/8 - 101*g1_2/40)
      + 65*g3_4/2 + 109*g2_4/8 - 523*g1_4/120
      + 21*g3_2*g2_2 + 77*g3_2*g1_2/15 - 3*g2_2*g1_2/40)
    t4 = g3_10/LOOP4 * 2472.28
    return t1+t2+t3+t4

def beta_lambda(y):
    (g1,g2,g3,yt,yb,ytau,lam,
     g1_2,g2_2,g3_2, g1_4,g2_4,g3_4,
     yt2,yb2,ytau2, yt4,yb4,ytau4) = _u(y)
    g1_6, g2_6 = g1_4*g1_2, g2_4*g2_2
    t1 = (1/LOOP1) * (
        lam*(12*lam + 6*yt2 + 6*yb2 + 2*ytau2 - 9*g2_2/2 - 9*g1_2/10)
      - 3*yt4 - 3*yb4 - ytau4
      + 9*g2_4/16 + 27*g1_4/400 + 9*g2_2*g1_2/40)
    t2 = (1/LOOP2) * (
        lam**2*(-156*lam - 72*yt2 - 72*yb2 - 24*ytau2 + 54*g2_2 + 54*g1_2/5)
      + lam*yt2*(-3*yt2/2 - 21*yb2 + 40*g3_2 + 45*g2_2/4 + 17*g1_2/4)
      + lam*yb2*(-3*yb2/2 + 40*g3_2 + 45*g2_2/4 + 5*g1_2/4)
      + lam*ytau2*(-ytau2/2 + 15*g2_2/4 + 15*g1_2/4)
      + lam*(-73*g2_4/16 + 1887*g1_4/400 + 117*g2_2*g1_2/40)
      + yt4*(15*yt2 - 3*yb2 - 16*g3_2 - 4*g1_2/5)
      + yt2*(-9*g2_4/8 - 171*g1_4/200 + 63*g2_2*g1_2/20)
      + yb4*(-3*yt2 + 15*yb2 - 16*g3_2 + 2*g1_2/5)
      + yb2*(-9*g2_4/8 + 9*g1_4/40 + 27*g2_2*g1_2/20)
      + ytau4*(5*ytau2 - 6*g1_2/5)
      + ytau2*(-3*g2_4/8 - 9*g1_4/8 + 33*g2_2*g1_2/20)
      + 305*g2_6/32 - 3411*g1_6/4000 - 289*g2_4*g1_2/160 - 1677*g2_2*g1_4/800)
    t3 = (1/LOOP3) * (
        lam**3*(6011.35*lam + 873*yt2 - 387.452*g2_2 - 77.490*g1_2)
      + lam**2*yt2*(1768.26*yt2 + 160.77*g3_2 - 359.539*g2_2 - 63.869*g1_2)
      + lam**2*(-790.28*g2_4 - 185.532*g1_4 - 316.64*g2_2*g1_2)
      + lam*yt4*(-223.382*yt2 - 662.866*g3_2 - 5.470*g2_2 - 21.015*g1_2)
      + lam*yt2*(356.968*g3_4 - 319.664*g2_4 - 74.8599*g1_4
                + 15.1443*g3_2*g2_2 + 17.454*g3_2*g1_2 + 5.615*g2_2*g1_2)
      + lam*g2_4*(-57.144*g3_2 + 865.483*g2_2 + 79.638*g1_2)
      + lam*g1_4*(-8.381*g3_2 + 61.753*g2_2 + 28.168*g1_2)
      + yt4*(-243.149*yt4 + 250.494*g3_2 + 74.138*g2_2 + 33.930*g1_2)
      + yt4*(-50.201*g3_4 + 15.884*g2_4 + 15.948*g1_4
            + 13.349*g3_2*g2_2 + 17.570*g3_2*g1_2 - 70.356*g2_2*g1_2)
      + yt2*g3_2*(16.464*g2_4 + 1.016*g1_4 + 11.386*g2_2*g1_2)
      + yt2*g2_4*(62.500*g2_2 + 13.041*g1_2)
      + yt2*g1_4*(10.627*g2_2 + 11.117*g1_2)
      + g3_2*(7.536*g2_6 + 0.663*g1_6 + 1.507*g2_4*g1_2 + 1.105*g2_2*g1_4)
      - 114.091*g2_6*g2_2 - 1.508*g1_6*g1_2 - 37.889*g2_4*g2_2*g1_2
      + 6.500*g2_4*g1_4 - 1.543*g2_2*g1_6)
    return t1+t2+t3

def beta_yt2(y):
    (g1,g2,g3,yt,yb,ytau,lam,
     g1_2,g2_2,g3_2, g1_4,g2_4,g3_4,
     yt2,yb2,ytau2, yt4,yb4,ytau4) = _u(y)
    g3_6 = g3_4*g3_2
    t1 = yt2/LOOP1 * (9*yt2/2 + 3*yb2/2 + ytau2 - 8*g3_2 - 9*g2_2/4 - 17*g1_2/20)
    t2 = yt2/LOOP2 * (
        yt2*(-12*yt2 - 11*yb2/4 - 9*ytau2/4 - 12*lam + 36*g3_2 + 225*g2_2/16 + 393*g1_2/80)
      + yb2*(-yb2/4 + 5*ytau2/4 + 4*g3_2 + 99*g2_2/16 + 7*g1_2/80)
      + ytau2*(-9*ytau2/4 + 15*g2_2/8 + 15*g1_2/8)
      + 6*lam**2 - 108*g3_4 - 23*g2_4/4 + 1187*g1_4/600
      + 9*g3_2*g2_2 + 19*g3_2*g1_2/15 - 9*g2_2*g1_2/20)
    t3 = yt2/LOOP3 * (
        yt2*(58.6028*yt2 + 198*lam - 157*g3_2 - 1593*g2_2/16 - 2437*g1_2/80)
      + lam*yt2*(15*lam/4 + 16*g3_2 - 135*g2_2/2 - 127*g1_2/10)
      + yt2*(363.764*g3_4 + 16.990*g2_4 - 24.422*g1_4
            + 48.370*g3_2*g2_2 + 18.074*g3_2*g1_2 + 34.829*g2_2*g1_2)
      + lam**2*(-36*lam + 45*g2_2 + 9*g1_2)
      + lam*(-171*g2_4/16 - 1089*g1_4/400 + 117*g2_2*g1_2/40)
      - 619.35*g3_6 + 169.829*g2_4*g2_2 + 16.099*g1_4*g1_2
      + 73.654*g3_4*g2_2 - 15.096*g3_4*g1_2
      - 21.072*g3_2*g2_4 - 22.319*g3_2*g1_4
      - 321*g3_2*g2_2*g1_2/20 - 4.743*g2_4*g1_2 - 4.442*g2_2*g1_4)
    return t1+t2+t3

def beta_yb2(y):
    (g1,g2,g3,yt,yb,ytau,lam,
     g1_2,g2_2,g3_2, g1_4,g2_4,g3_4,
     yt2,yb2,ytau2, yt4,yb4,ytau4) = _u(y)
    t1 = yb2/LOOP1 * (3*yt2/2 + 9*yb2/2 + ytau2 - 8*g3_2 - 9*g2_2/4 - g1_2/4)
    t2 = yb2/LOOP2 * (
        yt2*(-yt2/4 - 11*yb2/4 + 5*ytau2/4 + 4*g3_2 + 99*g2_2/16 + 91*g1_2/80)
      + yb2*(-12*yb2 - 9*ytau2/4 - 12*lam + 36*g3_2 + 225*g2_2/16 + 237*g1_2/80)
      + ytau2*(-9*ytau2/4 + 15*g2_2/8 + 15*g1_2/8)
      + 6*lam**2 - 108*g3_4 - 23*g2_4/4 - 127*g1_4/600
      + 9*g3_2*g2_2 + 31*g3_2*g1_2/15 - 27*g2_2*g1_2/20)
    return t1+t2

def beta_ytau2(y):
    (g1,g2,g3,yt,yb,ytau,lam,
     g1_2,g2_2,g3_2, g1_4,g2_4,g3_4,
     yt2,yb2,ytau2, yt4,yb4,ytau4) = _u(y)
    t1 = ytau2/LOOP1 * (3*yt2 + 3*yb2 + 5*ytau2/2 - 9*g2_2/4 - 9*g1_2/4)
    t2 = ytau2/LOOP2 * (
        6*lam**2 - 23*g2_4/4 + 1371*g1_4/200 + 27*g2_2*g1_2/20
      + yt2*(-27*yt2/4 + 3*yb2/2 - 27*ytau2/4 + 20*g3_2 + 45*g2_2/8 + 17*g1_2/8)
      + yb2*(-27*yb2/4 - 27*ytau2/4 + 20*g3_2 + 45*g2_2/8 + 5*g1_2/8)
      + ytau2*(-3*ytau2 - 12*lam + 165*g2_2/16 + 537*g1_2/80))
    return t1+t2

# ─── Effective coupling at high scale ─────────────────────────────────────────
def get_lambda_eff(y):
    g1, g2, g3, yt, yb, ytau, lam = y
    yt2 = yt**2; yt4 = yt2**2
    g2_2 = g2**2; g2_4 = g2_2**2
    g12 = g1**2 + g2_2; g12_2 = g12**2
    term_t = -3.0 * yt4 * (np.log(0.5 * yt2) - 1.5)
    term_W = 0.375 * g2_4 * (np.log(0.25 * g2_2) - 5.0/6.0)
    term_Z = 0.1875 * g12_2 * (np.log(0.25 * g12) - 5.0/6.0)
    return lam + (term_t + term_W + term_Z) / (16.0 * PI2)

# ─── 1-loop Coleman-Weinberg potential (near vev) ─────────────────────────────
def V_CW(phi, y_params):
    """Full 1-loop effective potential V(phi) = V_tree + V_1loop.
    Uses running couplings at scale mu = phi (or Mt if phi < Mt)."""
    g1, g2, g3, yt, yb, ytau, lam = y_params

    # Tree level: V = -mu^2/2 phi^2 + lam/4 phi^4
    # where mu^2 = lam * v^2 at the minimum
    mu2 = lam * v**2
    V_tree = -mu2 / 2.0 * phi**2 + lam / 4.0 * phi**4

    # 1-loop CW corrections (field-dependent masses)
    Mt2 = 0.5 * yt**2 * phi**2
    Mw2 = 0.25 * g2**2 * phi**2
    Mz2 = 0.25 * (g1**2 + g2**2) * phi**2
    
    # Renormalization scale = Mt (matching scale)
    mu_ren2 = (100.0)**2  # Mt = 100 GeV
    
    V_1loop = 0.0
    # Top quark (Nc=3, dof=12, fermion sign -)
    if Mt2 > 0:
        V_1loop += -12.0 * Mt2**2 * (np.log(Mt2/mu_ren2) - 1.5)
    # W boson (dof=6)
    if Mw2 > 0:
        V_1loop += 6.0 * Mw2**2 * (np.log(Mw2/mu_ren2) - 5.0/6.0)
    # Z boson (dof=3)
    if Mz2 > 0:
        V_1loop += 3.0 * Mz2**2 * (np.log(Mz2/mu_ren2) - 5.0/6.0)
    
    V_1loop /= (64.0 * PI2)
    
    return V_tree + V_1loop

# ─── RK4 step ────────────────────────────────────────────────────────────────
def dydt(y):
    g1, g2, g3, yt, yb, ytau, lam = y
    return np.array([
        0.5/g1   * beta_g1sq(y),
        0.5/g2   * beta_g2sq(y),
        0.5/g3   * beta_g3sq(y),
        0.5/yt   * beta_yt2(y),
        0.5/yb   * beta_yb2(y),
        0.5/ytau * beta_ytau2(y),
        beta_lambda(y)
    ])

def rk4_step(y, t, dt):
    k1 = dydt(y)
    k2 = dydt(y + 0.5*dt*k1)
    k3 = dydt(y + 0.5*dt*k2)
    k4 = dydt(y +     dt*k3)
    return y + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
Mt = 100.0   # GeV
Mh = 5.0     # GeV

y0 = get_nnlo_matching(Mh, Mt)
print("="*70)
print(f"  Point:  Mt = {Mt} GeV,  Mh = {Mh} GeV")
print(f"  NNLO matching at mu = Mt:")
print(f"    g1  = {y0[0]:.6f}")
print(f"    g2  = {y0[1]:.6f}")
print(f"    g3  = {y0[2]:.6f}")
print(f"    yt  = {y0[3]:.6f}")
print(f"    yb  = {y0[4]:.6f}")
print(f"    ytau= {y0[5]:.6f}")
lam_tree = Mh**2/(2*v**2)
print(f"    lam(Mt)  = {y0[6]:.6f}   (lam_tree = {lam_tree:.6f})")
print(f"    delta_lam = {y0[6] - lam_tree:.6f}  (NNLO threshold correction)")
print("="*70)

# Integrate from t0 = 2*ln(Mt) to tPl = 2*ln(M_Pl)
t0   = 2.0 * np.log(Mt)
tPl  = 2.0 * np.log(M_Pl)
dt   = 0.1

t_arr       = [t0]
mu_arr      = [Mt]
lam_arr     = [y0[6]]
lam_eff_arr = [get_lambda_eff(y0)]
yt_arr      = [y0[3]]
g1_arr      = [y0[0]]
g2_arr      = [y0[1]]
g3_arr      = [y0[2]]
# Store full coupling snapshots for interpolation
params_arr  = [y0.copy()]

y = y0.copy()
t = t0

min_lam_eff = lam_eff_arr[0]
t_min_lam   = t0
mu_inst     = -1.0
went_negative = min_lam_eff <= 0

if went_negative:
    mu_inst = np.exp(t / 2.0)

while t < tPl:
    h = min(dt, tPl - t)
    y = rk4_step(y, t, h)
    t += h

    mu = np.exp(t / 2.0)
    le = get_lambda_eff(y)

    t_arr.append(t)
    mu_arr.append(mu)
    lam_arr.append(y[6])
    lam_eff_arr.append(le)
    yt_arr.append(y[3])
    g1_arr.append(y[0])
    g2_arr.append(y[1])
    g3_arr.append(y[2])
    params_arr.append(y.copy())

    if le < min_lam_eff:
        min_lam_eff = le
        t_min_lam   = t
    if not went_negative and le <= 0:
        mu_inst = mu
        went_negative = True

    if abs(y[6]) > 4*pi or y[3] > 4*pi:
        print(f"  WARNING: Coupling blew up at mu = {mu:.2e} GeV, stopping.")
        break

mu_arr      = np.array(mu_arr)
lam_arr     = np.array(lam_arr)
lam_eff_arr = np.array(lam_eff_arr)
yt_arr      = np.array(yt_arr)
t_arr_np    = np.array(t_arr)
params_arr  = np.array(params_arr)

mu_min_lam  = np.exp(t_min_lam / 2.0)

print(f"\n  lam_eff first < 0 at  mu_1 = {mu_inst:.4e} GeV")
print(f"  Min lam_eff = {min_lam_eff:.6f}  at  mu = {mu_min_lam:.4e} GeV")

S_approx = 8.0 * PI2 / (3.0 * abs(min_lam_eff))
Tv       = 1.179e44 / v
S_thresh = 4.0 * np.log(Tv * mu_inst) if mu_inst > 0 else 1e10

print(f"\n  S_approx  = {S_approx:.2f}")
print(f"  S_thresh  = {S_thresh:.2f}")
print(f"  Classification (approx): {'Metastable' if S_approx > S_thresh else 'UNSTABLE'}")

# ─── Interpolation helpers ────────────────────────────────────────────────────
from scipy.interpolate import interp1d
lam_interp = interp1d(t_arr_np, lam_arr, kind='linear', fill_value='extrapolate')

def get_params_at_t(t_query):
    idx = np.searchsorted(t_arr_np, t_query) - 1
    idx = max(0, min(idx, len(t_arr_np)-2))
    f = (t_query - t_arr_np[idx]) / (t_arr_np[idx+1] - t_arr_np[idx])
    return params_arr[idx] + f * (params_arr[idx+1] - params_arr[idx])

def get_sm_lambda_at_phi(phi_dimless, mu_inst_val):
    h_val = phi_dimless * mu_inst_val
    h_rge = min(h_val, M_Pl)
    if h_rge <= v: return 0.0
    t_val = 2.0 * np.log(h_rge)
    t_val = np.clip(t_val, t_arr_np[0], t_arr_np[-1])
    return get_lambda_eff(get_params_at_t(t_val))

# ─── Bounce action (faster: N_QUAD=256 for scan) ─────────────────────────────
ALPHA_B = 4.0

def evaluate_action_at_R(R, mu_inst_val, N_q=256):
    if R <= 0: return 1e100, 0, 0
    mu_R = mu_inst_val / R
    if mu_R <= 0: return 1e100, 0, 0
    t_R = 2.0 * np.log(mu_R)
    if t_R < t_arr_np[0] or t_R > t_arr_np[-1]: return 1e100, 0, 0
    lam_R = float(lam_interp(t_R))
    if lam_R >= 0: return 1e100, 0, 0
    abs_lam = abs(lam_R)
    prefactor = np.sqrt(2.0 / abs_lam)
    kinetic = 16.0 * PI2 / (3.0 * abs_lam)
    if kinetic > 5e5: return kinetic, kinetic, 0

    du = 2.0 / N_q
    pot_int = 0.0
    for i in range(1, N_q):
        u = -1.0 + i * du
        if abs(u) >= 1.0: continue
        ratio = (1.0 + u) / (1.0 - u)
        e2x = ratio ** ALPHA_B
        e4x = e2x ** 2
        phi_x = prefactor * 2.0 / (R * (e2x + 1.0))
        lam_phi = get_sm_lambda_at_phi(phi_x, mu_inst_val)
        phi4 = phi_x**4
        V_SM = 0.25 * lam_phi * phi4
        V_6  = (1.0/6.0) * (phi_x * mu_inst_val / M_Pl)**2 * phi4
        V_x  = V_SM + V_6
        jacobian = ALPHA_B / (1.0 - u*u)
        f = 2.0 * PI2 * R**4 * e4x * V_x * jacobian
        weight = 4.0/3.0 if i%2 == 1 else 2.0/3.0
        pot_int += weight * f * du

    return kinetic + pot_int, kinetic, pot_int

# Golden section search
def find_min_action(mu_inst_val, t_min):
    mu_opt   = np.exp(t_min / 2.0)
    R_opt    = mu_inst_val / mu_opt
    logR_opt = np.log(R_opt)
    a, b = logR_opt - 8.0, logR_opt + 8.0
    tol = 1e-5
    invphi  = (np.sqrt(5) - 1) / 2
    invphi2 = (3 - np.sqrt(5)) / 2
    h = b - a
    n_iter = int(np.ceil(np.log(tol/h) / np.log(invphi)))
    c = a + invphi2 * h;  d = a + invphi * h
    yc = evaluate_action_at_R(np.exp(c), mu_inst_val)[0]
    yd = evaluate_action_at_R(np.exp(d), mu_inst_val)[0]
    for _ in range(n_iter):
        if yc < yd:
            b = d; d = c; yd = yc; h *= invphi
            c = a + invphi2 * h
            yc = evaluate_action_at_R(np.exp(c), mu_inst_val)[0]
        else:
            a = c; c = d; yc = yd; h *= invphi
            d = a + invphi * h
            yd = evaluate_action_at_R(np.exp(d), mu_inst_val)[0]
    R_min = np.exp(0.5*(a+b))
    S, Sk, Sp = evaluate_action_at_R(R_min, mu_inst_val, N_q=512)
    return S, R_min

if went_negative and mu_inst > 0:
    print("\n  Computing exact bounce action (numerical)...")
    S_exact, R_min = find_min_action(mu_inst, t_min_lam)
    print(f"  S_exact   = {S_exact:.2f}")
    print(f"  R_min     = {R_min:.4e}")
    print(f"  mu/R_min  = {mu_inst/R_min:.4e} GeV")
    print(f"  S_exact {'>' if S_exact > S_thresh else '<'} S_thresh  ->  "
          f"{'Metastable' if S_exact > S_thresh else 'UNSTABLE'}")
else:
    S_exact = None

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING  (3 rows x 2 cols)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle(rf'SM Vacuum Stability Analysis:  $M_t = {Mt:.0f}$ GeV,  $M_h = {Mh:.0f}$ GeV',
             fontsize=17, fontweight='bold', y=0.98)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (0,0): lambda running to Planck scale
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0, 0]
ax.plot(mu_arr, lam_arr,     'b-', lw=2, label=r'$\lambda(\mu)$ (MS-bar)')
ax.plot(mu_arr, lam_eff_arr, 'r-', lw=2, label=r'$\lambda_{\rm eff}(\mu)$')
ax.axhline(0, color='gray', ls='--', lw=0.8)
if mu_inst > 0:
    ax.axvline(mu_inst, color='orange', ls=':', lw=1.5,
               label=rf'$\mu_1 = {mu_inst:.1e}$ GeV')
ax.axvline(mu_min_lam, color='green', ls=':', lw=1.5,
           label=rf'$\mu_{{\min}} = {mu_min_lam:.1e}$ GeV')
ax.set_xscale('log')
ax.set_xlabel(r'$\mu$ [GeV]', fontsize=13)
ax.set_ylabel(r'$\lambda$', fontsize=14)
ax.set_title(r'Running of $\lambda$ and $\lambda_{\rm eff}$ to Planck Scale', fontsize=13)
ax.legend(fontsize=9, loc='best')
ax.set_xlim(Mt, M_Pl)
ax.grid(True, alpha=0.3)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (0,1): yt running to Planck scale
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0, 1]
ax.plot(mu_arr, yt_arr, 'purple', lw=2, label=r'$y_t(\mu)$')
ax.set_xscale('log')
ax.set_xlabel(r'$\mu$ [GeV]', fontsize=13)
ax.set_ylabel(r'$y_t$', fontsize=14)
ax.set_title(r'Running of top Yukawa $y_t$', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(Mt, M_Pl)
ax.grid(True, alpha=0.3)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (1,0): Effective potential at HIGH scales (full range to M_Pl)
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[1, 0]
phi_range_hi = np.logspace(np.log10(v*2), np.log10(M_Pl), 500)
V_eff_hi = np.zeros_like(phi_range_hi)
for i, phi_val in enumerate(phi_range_hi):
    t_phi = np.clip(2.0 * np.log(phi_val), t_arr_np[0], t_arr_np[-1])
    le = get_lambda_eff(get_params_at_t(t_phi))
    V_eff_hi[i] = 0.25 * le * phi_val**4

V_norm_hi = V_eff_hi / (v**4)
ax.plot(phi_range_hi, V_norm_hi, 'b-', lw=2)
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.fill_between(phi_range_hi, V_norm_hi, 0, where=(V_norm_hi < 0),
                color='red', alpha=0.15, label=r'$V_{\rm eff} < 0$ (unstable)')
ax.fill_between(phi_range_hi, V_norm_hi, 0, where=(V_norm_hi >= 0),
                color='blue', alpha=0.08)
ax.set_xscale('log')
ax.set_xlabel(r'$\phi$ [GeV]', fontsize=13)
ax.set_ylabel(r'$V_{\rm eff}(\phi) / v^4$', fontsize=13)
ax.set_title(r'RG-improved Potential $V_{\rm eff} = \frac{1}{4}\lambda_{\rm eff}(\mu\!=\!\phi)\,\phi^4$ (high scale)',
             fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(v*2, M_Pl)
ax.grid(True, alpha=0.3)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (1,1): Potential near the VEV well (log-log, up to 10^4 GeV)
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[1, 1]

# Compute the full 1-loop effective potential from phi ~ 1 GeV to 10^4 GeV
phi_low = np.logspace(-1, 4, 2000)  # 0.1 GeV to 10^4 GeV

# Tree-level potential with matched lambda
lam_matched = y0[6]
mu2_tree = lam_matched * v**2   # mu^2 parameter
V_tree_arr = -mu2_tree / 2.0 * phi_low**2 + lam_matched / 4.0 * phi_low**4

# Full 1-loop CW potential at the matching scale
V_full_arr = np.array([V_CW(phi, y0) for phi in phi_low])

# Also the RG-improved high-scale approximation for comparison
V_rgi_arr = np.zeros_like(phi_low)
for i, phi_val in enumerate(phi_low):
    t_phi = 2.0 * np.log(max(phi_val, Mt))
    t_phi = np.clip(t_phi, t_arr_np[0], t_arr_np[-1])
    le = get_lambda_eff(get_params_at_t(t_phi))
    V_rgi_arr[i] = 0.25 * le * phi_val**4

# Subtract V(0) = 0 for proper normalization  
V_at_vev_tree = -mu2_tree / 2.0 * v**2 + lam_matched / 4.0 * v**4
V_at_vev_full = V_CW(v, y0)

# Plot |V| on log scale, with color indicating sign
# Split into positive and negative parts
V_pos_tree = np.where(V_tree_arr > 0, V_tree_arr, np.nan)
V_neg_tree = np.where(V_tree_arr < 0, -V_tree_arr, np.nan)

V_pos_full = np.where(V_full_arr > 0, V_full_arr, np.nan)
V_neg_full = np.where(V_full_arr < 0, -V_full_arr, np.nan)

# Plot absolute value with sign indicated by solid/dashed
ax.plot(phi_low, V_pos_tree, 'b-', lw=1.5, alpha=0.5, label=r'$V_{\rm tree} > 0$')
ax.plot(phi_low, V_neg_tree, 'b--', lw=1.5, alpha=0.5, label=r'$|V_{\rm tree}| < 0$')
ax.plot(phi_low, V_pos_full, 'r-', lw=2, label=r'$V_{\rm 1\text{-}loop} > 0$')
ax.plot(phi_low, V_neg_full, 'r--', lw=2, label=r'$|V_{\rm 1\text{-}loop}|$ ($V<0$)')

# Mark the vev
ax.axvline(v, color='green', ls=':', lw=1.5, alpha=0.7, label=rf'$v = {v:.0f}$ GeV')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\phi$ [GeV]', fontsize=13)
ax.set_ylabel(r'$|V(\phi)|$ [GeV$^4$]', fontsize=13)
ax.set_title(r'Potential near EW scale (log-log, up to $10^4$ GeV)', fontsize=12)
ax.set_xlim(1, 1e4)
ax.set_ylim(bottom=1e-2)
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3, which='both')

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (2,0): V(phi) linear scale near vev to show the "well" shape
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[2, 0]

phi_lin = np.linspace(0, 2000, 3000)
V_tree_lin = -mu2_tree / 2.0 * phi_lin**2 + lam_matched / 4.0 * phi_lin**4
V_full_lin = np.array([V_CW(phi, y0) for phi in phi_lin])

# Normalize to v^4 for readability
ax.plot(phi_lin, V_tree_lin / v**4, 'b-', lw=1.5, alpha=0.6, label=r'Tree-level $V(\phi)$')
ax.plot(phi_lin, V_full_lin / v**4, 'r-', lw=2, label=r'1-loop $V(\phi)$')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.axvline(v, color='green', ls=':', lw=1.5, alpha=0.7, label=rf'$v = {v:.0f}$ GeV')

# Add annotation about lambda < 0
ax.annotate(rf'$\lambda(M_t) = {lam_matched:.5f}$' + '\n(negative: no stable minimum)',
            xy=(v, V_CW(v, y0)/v**4), fontsize=9,
            xytext=(600, 0.0001),
            arrowprops=dict(arrowstyle='->', color='darkred'),
            color='darkred', fontweight='bold')

ax.set_xlabel(r'$\phi$ [GeV]', fontsize=13)
ax.set_ylabel(r'$V(\phi) / v^4$', fontsize=13)
ax.set_title(r'Potential near EW scale (linear): No stable vacuum well', fontsize=12)
ax.set_xlim(0, 2000)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (2,1): Bounce action decomposition
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[2, 1]

if went_negative and mu_inst > 0:
    mu_opt = np.exp(t_min_lam / 2.0)
    R_opt  = mu_inst / mu_opt
    logR_center = np.log(R_opt)
    logR_scan = np.linspace(logR_center - 6, logR_center + 6, 30)
    S_kin_scan = []
    S_pot_scan = []
    S_tot_scan = []

    print("\n  Scanning bounce action over R (30 points)...")
    for j, logR in enumerate(logR_scan):
        R_val = np.exp(logR)
        S_t, S_k, S_p = evaluate_action_at_R(R_val, mu_inst, N_q=128)
        if S_t > 1e50:
            S_kin_scan.append(np.nan)
            S_pot_scan.append(np.nan)
            S_tot_scan.append(np.nan)
        else:
            S_kin_scan.append(S_k)
            S_pot_scan.append(S_p)
            S_tot_scan.append(S_t)
        if (j+1) % 10 == 0:
            print(f"    {j+1}/30 done")

    R_scan = np.exp(logR_scan)
    S_kin_scan = np.array(S_kin_scan)
    S_pot_scan = np.array(S_pot_scan)
    S_tot_scan = np.array(S_tot_scan)

    mask = np.isfinite(S_tot_scan)
    if np.any(mask):
        ax.plot(R_scan[mask], S_kin_scan[mask], 'g--', lw=1.5, label=r'$S_{\rm kin}$ (always $>0$)')
        ax.plot(R_scan[mask], S_pot_scan[mask], 'r--', lw=1.5, label=r'$S_{\rm pot}$ (can be $<0$)')
        ax.plot(R_scan[mask], S_tot_scan[mask], 'b-', lw=2.5, label=r'$S_{\rm total} = S_{\rm kin} + S_{\rm pot}$')
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        if S_exact is not None:
            ax.axhline(S_exact, color='magenta', ls=':', lw=1.5,
                       label=rf'$S_{{\rm min}} = {S_exact:.0f}$')
        ax.axhline(S_thresh, color='orange', ls=':', lw=1.5,
                   label=rf'$S_{{\rm thresh}} = {S_thresh:.0f}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'Bounce radius $R$', fontsize=13)
    ax.set_ylabel(r'$S_4(R)$', fontsize=14)
    ax.set_title('Bounce Action Decomposition', fontsize=13)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Potential stable\n(no bounce)',
            ha='center', va='center', transform=ax.transAxes, fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/negative_action_analysis.png', dpi=200, bbox_inches='tight')
print(f"\n  Plot saved to results/negative_action_analysis.png")
plt.close()
print("  Done!")
