"""
============================================================
 Step 1.3 — Balloon-Windkessel BOLD Model  [KAGGLE VERSION]

 Converts HRF-convolved neural signal y(t) into
 physiologically realistic fMRI BOLD signal using the
 Balloon-Windkessel hemodynamic model (Friston 2000).

 STATE VARIABLES:
   s = vasodilatory signal
   f = normalized cerebral blood flow (CBF)
   v = normalized cerebral blood volume (CBV)
   q = normalized deoxyhemoglobin content (dHb)

 BOLD SIGNAL:
   BOLD(t) = V0 · [k1(1-q) + k2(1-q/v) + k3(1-v)]

 PARAMETERS (from literature):
   kappa=0.65, gamma=0.41, tau=0.98, alpha=0.32
   E0=0.40, V0=0.02
   k1=7·E0=2.80, k2=2.0, k3=2·E0-0.2=0.60
   dt=0.05s → 600 steps × 0.05 = 30 seconds

 INPUT  : /kaggle/working/Neural_HRF_Data.csv
 OUTPUT : /kaggle/working/Neural_BOLD_Data.csv
   New columns: cbf_0..599, cbv_0..599, dhb_0..599, bold_0..599
   Total columns: 4,210 + 2,400 = 6,610
============================================================
"""

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import os, gc

CSV_IN  = "/kaggle/working/Neural_HRF_Data.csv"
CSV_OUT = "/kaggle/working/Neural_BOLD_Data.csv"

# ── BALLOON-WINDKESSEL PARAMETERS ───────────────────────────
kappa = 0.65    # signal decay rate
gamma = 0.41    # flow-dependent elimination
tau   = 0.98    # hemodynamic transit time (s)
alpha = 0.32    # Grubb's exponent
E0    = 0.40    # resting oxygen extraction fraction
V0    = 0.02    # resting blood volume fraction
k1    = 7.0 * E0           # 2.80
k2    = 2.0                # 2.00
k3    = 2.0 * E0 - 0.2    # 0.60

SEQ_LEN = 600
dt      = 0.05   # 50 ms per step → 30 seconds total

# Initial state: [s, f, v, q] = [0, 1, 1, 1]
S0, F0, V0_init, Q0 = 0.0, 1.0, 1.0, 1.0


def odes(state, u):
    """Balloon-Windkessel differential equations."""
    s, f, v, q = state
    f    = max(f, 1e-6); v = max(v, 1e-6); q = max(q, 1e-6)
    fout = max(v, 1e-6) ** (1.0 / alpha)
    E    = 1.0 - (1.0 - E0) ** (1.0 / max(f, 1e-6))
    ds   = u - s/kappa - (f-1.0)/gamma
    df   = s
    dv   = (f - fout) / tau
    dq   = (f*E/E0 - fout*q/v) / tau
    return np.array([ds, df, dv, dq])


def rk4(state, u):
    """4th-order Runge-Kutta integration step."""
    k1_ = odes(state,            u)
    k2_ = odes(state + dt/2*k1_, u)
    k3_ = odes(state + dt/2*k2_, u)
    k4_ = odes(state + dt  *k3_, u)
    return state + (dt/6.0)*(k1_ + 2*k2_ + 2*k3_ + k4_)


def run_balloon(u_arr):
    """Simulate Balloon-Windkessel model for one sample."""
    state = np.array([S0, F0, V0_init, Q0], dtype=float)
    cbf_out  = np.zeros(SEQ_LEN)
    cbv_out  = np.zeros(SEQ_LEN)
    dhb_out  = np.zeros(SEQ_LEN)
    bold_out = np.zeros(SEQ_LEN)

    for t in range(SEQ_LEN):
        state    = rk4(state, float(u_arr[t]))
        state[1] = max(state[1], 0.1)  # f floor
        state[2] = max(state[2], 0.1)  # v floor
        state[3] = max(state[3], 0.01) # q floor
        s,f,v,q  = state
        cbf_out[t]  = f
        cbv_out[t]  = v
        dhb_out[t]  = q
        bold_out[t] = V0*(k1*(1-q) + k2*(1-q/max(v,1e-6)) + k3*(1-v))

    return cbf_out, cbv_out, dhb_out, bold_out


print("=" * 60)
print("  Step 1.3 — Balloon-Windkessel  [KAGGLE]")
print(f"  dt={dt}s → {SEQ_LEN*dt:.0f}s total")
print(f"  kappa={kappa}  gamma={gamma}  tau={tau}")
print(f"  alpha={alpha}  E0={E0}  V0={V0}")
print(f"  k1={k1:.2f}  k2={k2:.2f}  k3={k3:.2f}")
print("=" * 60)

print(f"\n  Loading: {CSV_IN}")
df     = pd.read_csv(CSV_IN)
s_cols = [f"s_{i}" for i in range(SEQ_LEN)]
print(f"  Rows : {len(df):,}  |  Cols : {len(df.columns):,}")

# Run model
all_cbf, all_cbv, all_dhb, all_bold = [], [], [], []
print(f"\n  Running RK4 on {len(df):,} samples ...")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Balloon-Windkessel"):
    cbf,cbv,dhb,bold = run_balloon(row[s_cols].values.astype(float))
    all_cbf.append(cbf); all_cbv.append(cbv)
    all_dhb.append(dhb); all_bold.append(bold)

all_cbf  = np.array(all_cbf,  dtype=np.float32)
all_cbv  = np.array(all_cbv,  dtype=np.float32)
all_dhb  = np.array(all_dhb,  dtype=np.float32)
all_bold = np.array(all_bold, dtype=np.float32)

print(f"\n  Adding 2,400 columns ...")
for i in range(SEQ_LEN):
    df[f"cbf_{i}"]  = all_cbf[:, i]
    df[f"cbv_{i}"]  = all_cbv[:, i]
    df[f"dhb_{i}"]  = all_dhb[:, i]
    df[f"bold_{i}"] = all_bold[:, i]

print(f"\n  Saving → {CSV_OUT}")
df.to_csv(CSV_OUT, index=False)

reg = df['Class'].values==0; irr = df['Class'].values==1
rp  = all_bold[reg].max(axis=1).mean()
ip  = all_bold[irr].max(axis=1).mean()

print()
print("=" * 60)
print("  BOLD GENERATION COMPLETE")
print("=" * 60)
print(f"  Rows    : {len(df):,}   (expected 10,000)")
print(f"  Columns : {len(df.columns):,}   (expected 6,610)")
print(f"  BOLD max       : {all_bold.max():.6f}")
print(f"  Regular peak   : {rp:.6f}")
print(f"  Irregular peak : {ip:.6f}")
print(f"  Reg > Irr      : {'✅' if rp > ip else '⚠️'}")
print(f"  Output  : {CSV_OUT}")
print("=" * 60)
